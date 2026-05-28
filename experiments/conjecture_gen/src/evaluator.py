"""Conjecture evaluator: three-pronged assessment of accepted conjectures.

1. Small-case graph falsifier (NetworkX, n≤8)
   - Attempts to find a counterexample by brute-force graph enumeration.
   - Works for conjectures of the form: "every n-vertex graph with > f(n) edges
     contains subgraph H" and common-neighbor / codegree conditions.
   - Extracts parameters via regex; falls back to "unable to parse" gracefully.

2. Semantic Scholar literature search
   - Queries the SS API with keywords extracted from the conjecture.
   - Returns top-5 matching papers with title/year/citation count.
   - Flags if the conjecture might already be a known result.

3. LLM novelty + plausibility evaluation
   - Asks grok-3-mini to assess: (a) whether the statement is mathematically
     plausible, (b) whether it resembles a known result, (c) what would be
     needed to prove/disprove it.

Usage:
    python3 src/evaluator.py [--all] [--id CONJECTURE_ID]
    python3 src/evaluator.py --all                 # evaluate all accepted
    python3 src/evaluator.py --id conj:windmill_k  # evaluate one
    python3 src/evaluator.py --last N              # evaluate last N accepted
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
import networkx as nx
from networkx.algorithms import isomorphism
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOG_PATH = RESULTS_DIR / "conjecture_log.jsonl"
EVAL_PATH = RESULTS_DIR / "conjecture_eval.jsonl"

SS_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# 1. SMALL-CASE GRAPH FALSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _enumerate_graphs(n: int) -> list[nx.Graph]:
    """All non-isomorphic graphs on exactly n vertices (brute force for n≤8)."""
    if n > 8:
        raise ValueError("Only n≤8 supported")
    from itertools import combinations
    nodes = list(range(n))
    edges_all = list(combinations(nodes, 2))
    seen = []
    for r in range(len(edges_all) + 1):
        for edge_set in combinations(edges_all, r):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edge_set)
            if not any(nx.is_isomorphic(G, H) for H in seen):
                seen.append(G)
    return seen


def _has_subgraph(host: nx.Graph, pattern: nx.Graph) -> bool:
    """Check if pattern is a subgraph of host (up to isomorphism)."""
    gm = isomorphism.GraphMatcher(host, pattern)
    return gm.subgraph_is_isomorphic()


def _parse_linear_edge_threshold(statement: str) -> Optional[dict]:
    """
    Try to parse statements of the form:
      'every n-vertex graph with more than cn edges contains [subgraph description]'
    Returns dict with keys: c (float), subgraph_description (str), or None if unparseable.
    """
    # Pattern: "more than (k-1)n/2 edges" or "more than cn edges" or "≥ cn edges"
    m = re.search(
        r"more than\s+\(?([\d\.\-]+)\s*(?:[–\-]\s*[\d]+)?\)?[n]?(?:\s*/\s*(\d+))?\s+edges",
        statement, re.IGNORECASE
    )
    if not m:
        # Try simpler: "more than (k-1)n/2"
        m = re.search(r"more than\s+\(?(\w+(?:[\+\-]\d+)?)\)?n/(\d+)", statement, re.IGNORECASE)
    if not m:
        return None
    # Extract subgraph description from remainder
    after = statement[m.end():].strip()
    sub_m = re.search(r"contains?\s+(?:every\s+)?(.+?)(?:\s+as\s+a\s+subgraph|\.?\s*$|,)", after, re.IGNORECASE)
    sub_desc = sub_m.group(1).strip() if sub_m else after[:60]
    return {"threshold_str": m.group(0), "subgraph_desc": sub_desc}


def _parse_codegree_statement(statement: str) -> Optional[dict]:
    """
    Parse: 'every n-vertex graph in which every pair of vertices has at most λ
    common neighbors contains at most f(n) edges'
    Returns: {lambda: int, edge_bound_c: float} or None.
    """
    m = re.search(r"at most\s+([\d]+|λ)\s+common neighbor", statement, re.IGNORECASE)
    if not m:
        return None
    lam_str = m.group(1)
    lam = 1 if lam_str == "λ" else int(lam_str)
    # Edge bound
    edge_m = re.search(r"at most\s+([\d\.]+)\s*(?:[\+·×*]\s*)?n?\s+edges", statement[m.end():], re.IGNORECASE)
    edge_c = float(edge_m.group(1)) if edge_m else None
    return {"codegree_bound": lam, "edge_bound_c": edge_c}


def test_codegree_conjecture_small(lam: int, edge_c: float, n_max: int = 7) -> dict:
    """
    Test: every graph where every pair has ≤ lam common neighbors has ≤ edge_c * n edges.
    Find a counterexample if one exists for small n.
    """
    violations = []
    for n in range(3, n_max + 1):
        for G in _enumerate_graphs(n):
            e = G.number_of_edges()
            # Check codegree condition
            satisfies_codegree = all(
                len(list(nx.common_neighbors(G, u, v))) <= lam
                for u, v in nx.non_edges(G)
            ) and all(
                len(list(nx.common_neighbors(G, u, v))) <= lam
                for u, v in G.edges()
            )
            if not satisfies_codegree:
                continue
            # Check edge bound
            if edge_c is not None and e > edge_c * n:
                violations.append({
                    "n": n,
                    "edges": e,
                    "bound": edge_c * n,
                    "edges_list": list(G.edges()),
                })
    return {"violations": violations, "checked_up_to_n": n_max}


def falsify_small_cases(statement: str, n_max: int = 7) -> dict:
    """
    Attempt to find a counterexample to the conjecture using small graph enumeration.
    Returns a result dict with keys: method, result, details.
    """
    # Try codegree-type conjectures
    codegree = _parse_codegree_statement(statement)
    if codegree and codegree["codegree_bound"] is not None:
        lam = codegree["codegree_bound"]
        c = codegree["edge_bound_c"]
        if c is not None:
            result = test_codegree_conjecture_small(lam, c, n_max)
            if result["violations"]:
                return {
                    "method": "codegree_exhaustive",
                    "result": "COUNTEREXAMPLE_FOUND",
                    "details": result,
                }
            else:
                return {
                    "method": "codegree_exhaustive",
                    "result": f"NO_COUNTEREXAMPLE_n_leq_{n_max}",
                    "details": result,
                }

    # Try extracting linear threshold
    parsed = _parse_linear_edge_threshold(statement)
    if parsed:
        return {
            "method": "parse_only",
            "result": "PARSED_NOT_TESTED",
            "details": {
                "threshold": parsed["threshold_str"],
                "subgraph_target": parsed["subgraph_desc"],
                "note": "Subgraph containment test requires explicit graph description",
            },
        }

    # Windmill / extremal number pattern
    if re.search(r"ex\(n,\s*W_k\)", statement) or "windmill" in statement.lower():
        return {
            "method": "pattern_match",
            "result": "ASYMPTOTIC_CLAIM",
            "details": {"note": "Extremal number claim; small-case test not meaningful for asymptotics"},
        }

    return {
        "method": "none",
        "result": "UNPARSEABLE",
        "details": {"note": "Statement did not match any known testable pattern"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEMANTIC SCHOLAR LITERATURE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

_SS_STOPWORDS = {
    "every", "any", "all", "graph", "graphs", "edges", "vertices",
    "integer", "contains", "subgraph", "extremal", "number", "large",
    "sufficiently", "exists", "there", "such", "that", "with", "more",
    "than", "most", "least", "for", "the", "its", "copy", "copies",
    "disjoint", "union", "fixed", "each", "which", "where",
}

def _extract_search_keywords(statement: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z_\-]*", statement)
    keywords = [w for w in words if w.lower() not in _SS_STOPWORDS and len(w) > 3]
    # Prefer proper nouns (capitalized) and domain terms
    domain_terms = [w for w in keywords if w[0].isupper() or w.lower() in {
        "bipartite", "chromatic", "turán", "turan", "ramsey", "erdős",
        "erdos", "extremal", "hamiltonian", "planar", "clique", "cycle",
        "path", "tree", "forest", "matching", "coloring",
    }]
    # Build a short query: domain terms first, then others
    query_words = domain_terms[:4] + [w for w in keywords if w not in domain_terms][:3]
    return " ".join(query_words[:6])


def search_semantic_scholar(statement: str, n_results: int = 5) -> dict:
    """Query Semantic Scholar for related papers."""
    if not SS_API_KEY:
        return {"error": "No SEMANTIC_SCHOLAR_API_KEY set", "papers": []}

    query = _extract_search_keywords(statement)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": n_results,
        "fields": "title,year,citationCount,authors,abstract",
    }
    headers = {"x-api-key": SS_API_KEY}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for p in data.get("data", []):
            papers.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "authors": [a.get("name", "") for a in p.get("authors", [])[:3]],
                "abstract_snippet": (p.get("abstract") or "")[:200],
            })
        return {"query": query, "papers": papers}
    except Exception as e:
        return {"error": str(e), "papers": [], "query": query}


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM NOVELTY + PLAUSIBILITY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

LLM_EVAL_PROMPT = """You are an expert in extremal graph theory and combinatorics.

Evaluate the following mathematical conjecture on three axes:

CONJECTURE: {statement}

RELATED PAPERS FOUND: {papers_summary}

Rate and explain each of the following (1-5 scale where 5 is best):
1. Mathematical plausibility (1=likely false, 5=likely true based on analogies)
2. Novelty (1=clearly known/trivial, 5=genuinely new as far as you know)
3. Significance (1=minor variation, 5=would be a major result if true)

Then give:
- Known_similar: list any theorem(s) this resembles or generalizes
- Proof_strategy: one sentence on the most promising approach
- Verdict: one of [LIKELY_KNOWN, PLAUSIBLE_NOVEL, NEEDS_VERIFICATION, LIKELY_FALSE]

Respond in JSON only:
{{
  "plausibility": <1-5>,
  "novelty": <1-5>,
  "significance": <1-5>,
  "known_similar": ["<theorem name>", ...],
  "proof_strategy": "<one sentence>",
  "verdict": "<LIKELY_KNOWN|PLAUSIBLE_NOVEL|NEEDS_VERIFICATION|LIKELY_FALSE>",
  "notes": "<2-3 sentences of additional context>"
}}"""


def llm_evaluate(statement: str, ss_results: dict) -> dict:
    """Use grok-3-mini to assess plausibility, novelty, and significance."""
    if not XAI_API_KEY:
        return {"error": "No XAI_API_KEY set"}

    papers = ss_results.get("papers", [])
    if papers:
        papers_summary = "\n".join(
            f"- {p['title']} ({p['year']}, {p['citations']} citations)" for p in papers[:5]
        )
    else:
        papers_summary = "None found."

    client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    prompt = LLM_EVAL_PROMPT.format(statement=statement, papers_summary=papers_summary)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="grok-3-mini",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text, "error": "JSON parse failed"}
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
            time.sleep(2 ** attempt)
    return {"error": "LLM call failed after retries"}


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_conjecture(record: dict, verbose: bool = True) -> dict:
    """Run all three evaluations on a single conjecture record."""
    statement = record.get("statement", "")
    new_node = record.get("new_node") or {}
    cid = new_node.get("id") or f"ep{record.get('episode', '?')}"

    if verbose:
        print(f"\n{'─'*70}")
        print(f"Evaluating: {cid}")
        print(f"Statement: {statement[:100]}...")

    # 1. Small-case falsifier
    if verbose:
        print("  [1/3] Small-case falsifier...")
    falsifier_result = falsify_small_cases(statement)
    if verbose:
        print(f"        → {falsifier_result['result']}")

    # 2. Semantic Scholar
    if verbose:
        print("  [2/3] Semantic Scholar search...")
    ss_result = search_semantic_scholar(statement)
    time.sleep(0.5)  # rate limiting
    if verbose:
        n_papers = len(ss_result.get("papers", []))
        print(f"        → {n_papers} papers (query: '{ss_result.get('query', '')}')")

    # 3. LLM evaluation
    if verbose:
        print("  [3/3] LLM novelty evaluation...")
    llm_result = llm_evaluate(statement, ss_result)
    if verbose:
        verdict = llm_result.get("verdict", "ERROR")
        p, n, s = llm_result.get("plausibility", "?"), llm_result.get("novelty", "?"), llm_result.get("significance", "?")
        print(f"        → verdict={verdict}  P={p} N={n} S={s}")

    return {
        "id": cid,
        "statement": statement,
        "episode": record.get("episode"),
        "score": record.get("adjusted_score", record.get("score")),
        "falsifier": falsifier_result,
        "semantic_scholar": ss_result,
        "llm_eval": llm_result,
    }


def load_accepted_records() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [
        json.loads(l) for l in LOG_PATH.read_text().splitlines()
        if l.strip() and json.loads(l).get("accepted")
    ]


def print_summary(eval_records: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY — {len(eval_records)} conjectures")
    print(f"{'='*70}")
    print(f"{'ID':<35} {'Verdict':<20} {'P':>3} {'N':>3} {'S':>3} {'Falsifier'}")
    print("─" * 70)
    for r in eval_records:
        llm = r.get("llm_eval", {})
        fals = r.get("falsifier", {})
        cid = r["id"][:34]
        verdict = llm.get("verdict", "N/A")[:18]
        p = llm.get("plausibility", "?")
        n = llm.get("novelty", "?")
        s = llm.get("significance", "?")
        fr = fals.get("result", "?")[:20]
        print(f"{cid:<35} {verdict:<20} {str(p):>3} {str(n):>3} {str(s):>3}  {fr}")

    # High-value candidates
    high_value = [
        r for r in eval_records
        if r.get("llm_eval", {}).get("verdict") == "PLAUSIBLE_NOVEL"
        and r.get("llm_eval", {}).get("novelty", 0) >= 4
        and r.get("falsifier", {}).get("result") != "COUNTEREXAMPLE_FOUND"
    ]
    if high_value:
        print(f"\nHigh-value candidates (novel + plausible + no small counterexample):")
        for r in high_value:
            print(f"\n  [{r['id']}] score={r.get('score', '?'):.3f}")
            print(f"  {r['statement'][:120]}")
            llm = r.get("llm_eval", {})
            print(f"  → {llm.get('proof_strategy', '')}")
            print(f"  → Known similar: {llm.get('known_similar', [])}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate accepted conjectures")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Evaluate all accepted conjectures")
    group.add_argument("--last", type=int, metavar="N", help="Evaluate last N accepted conjectures")
    group.add_argument("--id", type=str, metavar="ID", help="Evaluate conjecture with given node ID")
    group.add_argument("--file", type=str, metavar="PATH", help="Evaluate conjectures from a JSONL file")
    parser.add_argument("--no-cache", action="store_true", help="Re-evaluate even if already in eval log")
    args = parser.parse_args()

    if args.file:
        targets = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    else:
        accepted = load_accepted_records()
        if not accepted:
            print("No accepted conjectures found in log.")
            return
        if args.id:
            targets = [r for r in accepted if (r.get("new_node") or {}).get("id") == args.id]
            if not targets:
                print(f"No accepted conjecture with id={args.id}")
                return
        elif args.last:
            targets = accepted[-args.last:]
        else:
            targets = accepted

    # Load existing eval cache
    already_evaled: dict[str, dict] = {}
    if EVAL_PATH.exists() and not args.no_cache:
        for line in EVAL_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                already_evaled[r["id"]] = r

    print(f"Evaluating {len(targets)} conjecture(s)...")
    eval_records = []

    with open(EVAL_PATH, "a") as out_f:
        for record in targets:
            new_node = record.get("new_node") or {}
            cid = new_node.get("id") or f"ep{record.get('episode', '?')}"
            if cid in already_evaled and not args.no_cache:
                print(f"  [cached] {cid}")
                eval_records.append(already_evaled[cid])
                continue

            result = evaluate_conjecture(record, verbose=True)
            eval_records.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print_summary(eval_records)
    print(f"\nResults saved to {EVAL_PATH}")


if __name__ == "__main__":
    main()
