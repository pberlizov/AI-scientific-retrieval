"""Real LLM-based conjecture agent using grok-3-mini via xAI API.

The agent receives a formatted view of the current knowledge graph and proposes
new edges (mathematical relationships) between existing or new nodes. Proposals
are returned as structured JSON rather than Lean 4 syntax — formalization is a
later step.
"""

import json
import os
import re
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from knowledge_graph.graph import KnowledgeGraph, NodeType, EdgeType

EDGE_TYPES = [e.name for e in EdgeType]
NODE_TYPES = [n.name for n in NodeType]

SYSTEM_PROMPT = """You are a mathematical conjecture generator specializing in extremal graph theory.

You are given a knowledge graph of theorems, objects, quantities, and their relationships.
Your task is to propose a new mathematical conjecture — a plausible, non-trivial statement
that EITHER:
  (a) Introduces a new theorem/conjecture node and connects it to existing nodes, OR
  (b) Proposes a missing edge between two existing nodes that would be mathematically interesting.

Respond with a single JSON object and nothing else:
{
  "statement": "<precise mathematical statement in plain English>",
  "new_node": {
    "id": "<snake_case_id prefixed with conj: or thm:>",
    "type": "<THEOREM or CONJECTURE>",
    "name": "<human-readable name>"
  },
  "proposed_edges": [
    {
      "source_id": "<existing or new node id>",
      "target_id": "<existing or new node id>",
      "edge_type": "<one of: IMPLIES, GENERALIZES, TECHNIQUE_USED, COUNTEREXAMPLE, BOUNDS, HAS_CONDITION, SPECIAL_CASE>",
      "rationale": "<one sentence explaining why this edge is mathematically justified>"
    }
  ],
  "why_interesting": "<one sentence: what makes this non-trivial or surprising>"
}

Rules:
- Every id in proposed_edges must either exist in the graph or be the new_node id.
- new_node may be null if you are only proposing a new edge between existing nodes.
- Prefer conjectures at the boundary of known results — generalize, tighten bounds, or bridge disconnected areas.
- Do NOT propose trivial consequences (e.g. that a special case of K_n is a graph).
- Do NOT propose things already known as standard theorems unless connecting them in a new way.
"""


def format_kg_context(kg: KnowledgeGraph) -> str:
    """Render the knowledge graph as a compact text block for the LLM prompt."""
    lines = ["=== KNOWLEDGE GRAPH ===", ""]

    lines.append("NODES:")
    for node_id, data in kg.graph.nodes(data=True):
        ntype = data.get("type")
        type_name = ntype.name if ntype else "UNKNOWN"
        name = data.get("name", node_id)
        stmt = data.get("statement", data.get("description", ""))
        line = f"  [{type_name}] {node_id} — {name}"
        if stmt:
            line += f"\n    {stmt}"
        lines.append(line)

    lines.append("")
    lines.append("EDGES:")
    seen = set()
    for u, v, data in kg.graph.edges(data=True):
        etype = data.get("etype")
        type_name = etype.name if etype else "UNKNOWN"
        key = (u, type_name, v)
        if key not in seen:
            seen.add(key)
            desc = data.get("description", data.get("rationale", ""))
            line = f"  {u} --[{type_name}]--> {v}"
            if desc:
                line += f"  ({desc})"
            lines.append(line)

    return "\n".join(lines)


def _parse_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, tolerating markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class LLMConjectureAgent:
    def __init__(self, model: str = "grok-3-mini"):
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise SystemExit("XAI_API_KEY not set.")
        self.client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        self.model = model
        self.history: list[dict] = []  # track proposals across episodes

    def generate_conjecture(self, kg: KnowledgeGraph) -> Optional[dict]:
        """
        Returns a parsed proposal dict or None on failure.
        Dict has keys: statement, new_node (nullable), proposed_edges, why_interesting.
        """
        context = format_kg_context(kg)

        # Include a few recent proposals to encourage novelty
        recent = ""
        if self.history:
            recent = "\n\nPREVIOUS PROPOSALS (do not repeat these):\n"
            for prev in self.history[-5:]:
                recent += f"  - {prev.get('statement', '?')}\n"

        user_content = context + recent + "\n\nPropose one new conjecture or relationship."

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=600,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                proposal = _parse_response(raw)
                if proposal:
                    self.history.append(proposal)
                    return proposal
                print(f"  [LLMAgent] Parse failed on attempt {attempt+1}. Raw: {raw[:200]}")
            except Exception as e:
                print(f"  [LLMAgent] API error on attempt {attempt+1}: {e}")

        return None
