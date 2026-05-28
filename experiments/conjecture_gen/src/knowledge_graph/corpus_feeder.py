"""Pull math-adjacent claims from the scientific-discovery corpus DB and add
them as nodes to the knowledge graph.

Uses grok-3-mini to classify each claim as relevant/irrelevant and extract
the node type and any connections to existing KG nodes.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(__file__))
from graph import KnowledgeGraph, NodeType, EdgeType

DB_PATH = Path(__file__).parents[4] / "data" / "db" / "knowledge.db"

_EDGE_TYPE_MAP = {e.name: e for e in EdgeType}
_NODE_TYPE_MAP = {n.name: n for n in NodeType}

CLASSIFY_PROMPT = """You are filtering claims from academic papers for relevance to a knowledge graph
about extremal graph theory and combinatorics.

Given a claim from an academic paper, decide if it is a theorem, lemma, bound, or conjecture
that could plausibly connect to extremal graph theory, combinatorics, linear algebra applied to
graphs, Ramsey theory, or graph algorithms.

Respond with a JSON object:
{
  "relevant": true/false,
  "reason": "<one sentence>",
  "node": {
    "id": "<snake_case prefixed with thm:, conj:, tech:, or obj:>",
    "type": "<THEOREM, CONJECTURE, TECHNIQUE, or OBJECT>",
    "name": "<concise human name>",
    "statement": "<the claim, cleaned up>"
  },
  "connects_to": [
    {
      "existing_id": "<id of node already in the graph, if any>",
      "edge_type": "<IMPLIES, GENERALIZES, TECHNIQUE_USED, BOUNDS, HAS_CONDITION, SPECIAL_CASE, or COUNTEREXAMPLE>",
      "rationale": "<one sentence>"
    }
  ]
}

If not relevant, set relevant=false and omit node/connects_to.
Only populate connects_to if you are confident the connection is mathematically justified.
"""


def fetch_candidate_claims(limit: int = 200) -> list[dict]:
    """Pull claims likely to be mathematical in nature."""
    conn = sqlite3.connect(DB_PATH)
    keywords = [
        "theorem", "lemma", "bound", "conjecture", "inequality",
        "graph", "clique", "chromatic", "coloring", "matching",
        "matrix", "eigenvalue", "spectral", "polynomial", "sparse",
        "complexity", "algorithm", "combinat", "probability bound",
        "information", "entropy", "linear algebra", "random graph",
    ]
    like_clauses = " OR ".join(f"c.statement LIKE '%{k}%'" for k in keywords)
    rows = conn.execute(f"""
        SELECT c.id, c.statement, d.title, d.year
        FROM claims c
        JOIN documents d ON c.doc_id = d.id
        WHERE ({like_clauses})
          AND c.statement IS NOT NULL
          AND length(c.statement) > 60
          AND length(c.statement) < 600
        ORDER BY d.year ASC
        LIMIT {limit}
    """).fetchall()
    conn.close()
    return [{"claim_id": r[0], "statement": r[1], "paper": r[2], "year": r[3]} for r in rows]


def classify_claim(client: OpenAI, claim: dict, existing_ids: set[str]) -> Optional[dict]:
    """Ask the LLM whether this claim is relevant and how to add it."""
    existing_sample = ", ".join(sorted(existing_ids)[:40])
    user_msg = (
        f"Paper: {claim['paper']} ({claim['year']})\n"
        f"Claim: {claim['statement']}\n\n"
        f"Existing node IDs (sample): {existing_sample}"
    )
    try:
        resp = client.chat.completions.create(
            model="grok-3-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content or ""
        # Strip markdown fences
        import re
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
        return json.loads(raw)
    except Exception as e:
        return None


def feed_corpus_into_kg(kg: KnowledgeGraph, max_claims: int = 200,
                        min_confidence: float = 0.0) -> int:
    """
    Fetch claims from the corpus DB, classify them, and add relevant ones to kg.
    Returns the number of nodes added.
    """
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY not set.")
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    claims = fetch_candidate_claims(limit=max_claims)
    print(f"Fetched {len(claims)} candidate claims from corpus.")

    existing_ids = set(kg.graph.nodes())
    added = 0

    for i, claim in enumerate(claims):
        result = classify_claim(client, claim, existing_ids)
        if not result or not result.get("relevant"):
            continue

        node = result.get("node", {})
        nid = node.get("id", "")
        ntype_name = node.get("type", "THEOREM")
        ntype = _NODE_TYPE_MAP.get(ntype_name, NodeType.THEOREM)

        if not nid or nid in kg.graph:
            continue

        kg.add_node(nid, ntype, {
            "name": node.get("name", nid),
            "statement": node.get("statement", claim["statement"]),
            "source": claim["paper"],
            "year": claim["year"],
        })
        existing_ids.add(nid)
        added += 1
        print(f"  [{i+1}] Added [{ntype_name}] {nid}: {node.get('name', '')}")

        for conn in result.get("connects_to", []):
            target = conn.get("existing_id", "")
            etype_name = conn.get("edge_type", "")
            etype = _EDGE_TYPE_MAP.get(etype_name)
            if target in kg.graph and etype:
                try:
                    kg.add_edge(nid, target, etype, {"rationale": conn.get("rationale", "")})
                    print(f"    -> [{etype_name}] {nid} → {target}")
                except ValueError:
                    pass

    print(f"\nCorpus feeder done: {added} nodes added.")
    return added


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from knowledge_graph.seed import generate_seed_graph

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-claims", type=int, default=100)
    args = parser.parse_args()

    kg = generate_seed_graph()
    print(f"Seed graph: {kg.graph.number_of_nodes()} nodes")
    feed_corpus_into_kg(kg, max_claims=args.max_claims)
    print(f"\nFinal graph: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
    print(kg.summary())
