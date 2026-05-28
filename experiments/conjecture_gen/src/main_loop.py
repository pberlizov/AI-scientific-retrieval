"""Main conjecture generation loop.

Each episode:
  1. Format the current knowledge graph as LLM context.
  2. Ask the LLM agent to propose a new conjecture or relationship.
  3. Score the proposal with the interestingness reward model.
  4. If score is above threshold, add the new node/edges to the KG.
  5. Persist the proposal to conjecture_log.jsonl.

Usage:
    python3 src/main_loop.py [--episodes N] [--threshold 0.3]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from knowledge_graph.extractor import build_graph_from_mathlib
from knowledge_graph.graph import KnowledgeGraph, NodeType, EdgeType
from knowledge_graph.seed import generate_seed_graph
from reward_model import compute_interestingness
from reward_model.surprise import train_gnn
from generator.llm_agent import LLMConjectureAgent

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOG_PATH = RESULTS_DIR / "conjecture_log.jsonl"

_EDGE_TYPE_MAP = {e.name: e for e in EdgeType}
_NODE_TYPE_MAP = {n.name: n for n in NodeType}

_DIVERSITY_WINDOW = 5  # matches agent's history window

# Content words that define a "family" — 2+ shared with last 3 accepted → veto
_FAMILY_STOPWORDS = {
    "every", "for", "all", "sufficiently", "large", "with", "more", "than",
    "edges", "graph", "graphs", "n-vertex", "integer", "and", "the", "a",
    "an", "in", "of", "is", "are", "at", "to", "has", "have", "least",
    "most", "that", "its", "unique", "uniquely", "n", "k", "t", "s", "m",
    "contains", "copy", "where", "which", "each", "such", "also",
}


def _statement_keywords(statement: str) -> set[str]:
    """Extract meaningful content words from a conjecture statement."""
    import re
    words = re.findall(r"[a-z][a-z_\-]*", statement.lower())
    return {w for w in words if w not in _FAMILY_STOPWORDS and len(w) > 3}


def _same_family_veto(proposal: dict, accepted_records: list[dict], window: int = 3) -> bool:
    """
    Return True if the proposal shares ≥3 content keywords with every one of
    the last `window` accepted proposals — i.e., it's in the same family.
    """
    recent = [r for r in accepted_records if r.get("accepted")][-window:]
    if len(recent) < window:
        return False
    kw = _statement_keywords(proposal.get("statement", ""))
    if not kw:
        return False
    for r in recent:
        overlap = kw & _statement_keywords(r.get("statement", ""))
        if len(overlap) < 3:
            return False  # at least one recent proposal is in a different family
    return True  # all recent proposals share ≥3 keywords → same family


def _recent_anchor_nodes(records: list[dict], window: int = _DIVERSITY_WINDOW) -> set[str]:
    """Return node IDs touched by the last `window` accepted proposals."""
    accepted = [r for r in records if r.get("accepted")][-window:]
    nodes: set[str] = set()
    for r in accepted:
        nn = r.get("new_node") or {}
        if nn.get("id"):
            nodes.add(nn["id"])
        for edge in r.get("proposed_edges", []):
            nodes.add(edge.get("source_id", ""))
            nodes.add(edge.get("target_id", ""))
    nodes.discard("")
    return nodes


def _diversity_multiplier(proposal: dict, recent_nodes: set[str]) -> float:
    """
    Score in [0.2, 1.0]: penalises proposals that only connect nodes seen recently.
    1.0 = every proposed edge touches a node outside the recent window.
    0.2 = every proposed edge is entirely within the recent window.
    """
    if not recent_nodes:
        return 1.0
    edges = proposal.get("proposed_edges", [])
    if not edges:
        return 1.0
    novel = sum(
        1 for e in edges
        if e.get("source_id") not in recent_nodes or e.get("target_id") not in recent_nodes
    )
    return 0.2 + 0.8 * (novel / len(edges))


def apply_proposal(kg: KnowledgeGraph, proposal: dict) -> list[tuple[str, str, EdgeType]]:
    """
    Add new node (if any) and proposed edges to the KG.
    Returns the list of (source, target, EdgeType) edges that were successfully added.
    """
    new_node = proposal.get("new_node")
    if new_node:
        node_id = new_node.get("id", "")
        node_type_name = new_node.get("type", "CONJECTURE")
        node_type = _NODE_TYPE_MAP.get(node_type_name, NodeType.CONJECTURE)
        if node_id and node_id not in kg.graph:
            kg.add_node(node_id, node_type, {
                "name": new_node.get("name", node_id),
                "statement": proposal.get("statement", ""),
            })

    added = []
    for edge in proposal.get("proposed_edges", []):
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        etype_name = edge.get("edge_type", "")
        etype = _EDGE_TYPE_MAP.get(etype_name)

        if not src or not tgt or not etype:
            print(f"  [apply] Skipping malformed edge: {edge}")
            continue
        if src not in kg.graph:
            print(f"  [apply] Unknown source node: {src}")
            continue
        if tgt not in kg.graph:
            print(f"  [apply] Unknown target node: {tgt}")
            continue

        try:
            kg.add_edge(src, tgt, etype, {"rationale": edge.get("rationale", "")})
            added.append((src, tgt, etype))
        except ValueError as e:
            print(f"  [apply] Edge error: {e}")

    return added


def run_loop(episodes: int = 10, threshold: float = 0.5, use_mathlib: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the base knowledge graph
    if use_mathlib:
        data_dir = Path(__file__).parent.parent / "data" / "mathlib4"
        print(f"Building KG from Mathlib4 ({data_dir})...")
        kg = build_graph_from_mathlib(str(data_dir))
    else:
        print("Using seed knowledge graph (Mantel, Turán, Erdős–Stone)...")
        kg = generate_seed_graph()

    print(f"Base graph: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")

    print("Training GNN for surprise scoring...")
    gnn_model = train_gnn(kg, epochs=100)

    agent = LLMConjectureAgent()

    print(f"\nStarting conjecture loop ({episodes} episodes, accept threshold={threshold})\n")

    # Load existing log to seed the diversity tracker
    accepted_records: list[dict] = []
    if LOG_PATH.exists():
        accepted_records = [
            json.loads(l) for l in LOG_PATH.read_text().splitlines()
            if l.strip() and json.loads(l).get("accepted")
        ]

    with open(LOG_PATH, "a") as log_f:
        for ep in range(1, episodes + 1):
            print(f"{'='*60}")
            print(f"Episode {ep}/{episodes}")
            print(f"  Graph state: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")

            proposal = agent.generate_conjecture(kg)
            if not proposal:
                print("  Agent failed to produce a proposal. Skipping.")
                continue

            print(f"\n  Statement: {proposal.get('statement', '?')}")
            print(f"  Why interesting: {proposal.get('why_interesting', '?')}")

            # Score before applying (so KG state reflects pre-proposal graph)
            edges_to_score = proposal.get("proposed_edges", [])
            if not edges_to_score:
                print("  No edges proposed. Skipping.")
                continue

            # Temporarily add new node (if any) for scoring, but don't commit edges yet
            new_node = proposal.get("new_node")
            temp_node_added = False
            if new_node:
                nid = new_node.get("id", "")
                if nid and nid not in kg.graph:
                    nt = _NODE_TYPE_MAP.get(new_node.get("type", "CONJECTURE"), NodeType.CONJECTURE)
                    kg.add_node(nid, nt, {"name": new_node.get("name", nid), "statement": proposal.get("statement", "")})
                    temp_node_added = True

            scores = []
            for edge in edges_to_score:
                src = edge.get("source_id", "")
                tgt = edge.get("target_id", "")
                etype_name = edge.get("edge_type", "")
                etype = _EDGE_TYPE_MAP.get(etype_name)
                if src in kg.graph and tgt in kg.graph and etype:
                    s = compute_interestingness(kg, gnn_model, src, tgt, etype)
                    scores.append(s)
                    print(f"  Score [{etype_name}] {src} -> {tgt}: {s:.3f}")

            mean_score = sum(scores) / len(scores) if scores else 0.0

            # Same-family veto: hard reject if statement shares a family with last 3 accepted
            if _same_family_veto(proposal, accepted_records):
                print("  Same-family veto — asking agent to explore a different area.")
                if temp_node_added and new_node:
                    nid = new_node.get("id", "")
                    if nid and nid in kg.graph:
                        kg.graph.remove_node(nid)
                record = {**proposal, "score": mean_score, "diversity": 0.0,
                          "adjusted_score": 0.0, "accepted": False,
                          "reject_reason": "same_family", "episode": ep}
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
                continue

            # Diversity penalty: discourage re-exploiting the same node cluster
            recent_nodes = _recent_anchor_nodes(accepted_records)
            div_mult = _diversity_multiplier(proposal, recent_nodes)
            adjusted_score = mean_score * div_mult
            print(f"  Mean interestingness: {mean_score:.3f}  diversity: {div_mult:.2f}  "
                  f"adjusted: {adjusted_score:.3f}  (threshold: {threshold})")

            # If below threshold, remove temp node and skip
            if adjusted_score < threshold:
                print("  Below threshold — rejected.")
                if temp_node_added and new_node:
                    nid = new_node.get("id", "")
                    if nid and nid in kg.graph:
                        kg.graph.remove_node(nid)
                record = {**proposal, "score": mean_score, "diversity": div_mult,
                          "adjusted_score": adjusted_score, "accepted": False, "episode": ep}
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
                continue

            # Apply edges to the graph
            added_edges = apply_proposal(kg, proposal)
            print(f"  Accepted! Added {len(added_edges)} edge(s) to the graph.")

            # Retrain GNN on the updated graph
            gnn_model = train_gnn(kg, epochs=50)

            record = {**proposal, "score": mean_score, "diversity": div_mult,
                      "adjusted_score": adjusted_score, "accepted": True, "episode": ep,
                      "edges_added": len(added_edges)}
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            accepted_records.append(record)

    print(f"\nDone. {episodes} episodes complete.")
    print(f"Results logged to {LOG_PATH}")
    _print_summary()


def _print_summary():
    if not LOG_PATH.exists():
        return
    records = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
    accepted = [r for r in records if r.get("accepted")]
    print(f"\n{'='*60}")
    print(f"CONJECTURE LOG SUMMARY — {len(accepted)}/{len(records)} accepted")
    for r in accepted:
        print(f"\n  [ep {r['episode']}] score={r['score']:.3f}")
        print(f"  {r.get('statement', '?')}")
        print(f"  {r.get('why_interesting', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mathlib", action="store_true",
                        help="Build KG from Mathlib4 files instead of seed graph")
    args = parser.parse_args()
    run_loop(episodes=args.episodes, threshold=args.threshold, use_mathlib=args.mathlib)
