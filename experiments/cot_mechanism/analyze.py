"""Analyze evaluation results and print summary table.

Usage:
    python experiments/cot_mechanism/analyze.py --results experiments/cot_mechanism/results/
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CONDITIONS = ["direct", "scribe", "usurer", "broken"]


def load_results(results_dir: Path, model_slug: str | None = None) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*_results.jsonl")):
        if model_slug and model_slug not in path.name:
            continue
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)

    summary = {}
    for cond, results in by_condition.items():
        n = len(results)
        correct = sum(1 for r in results if r["correct"])
        avg_trace_len = sum(r["trace_length"] for r in results) / n if n else 0
        summary[cond] = {
            "n": n,
            "accuracy": correct / n if n else 0,
            "correct": correct,
            "avg_trace_chars": avg_trace_len,
        }
    return summary


def interpret(summary: dict) -> str:
    direct_acc = summary.get("direct", {}).get("accuracy", 0)
    scribe_acc = summary.get("scribe", {}).get("accuracy", 0)
    broken_acc = summary.get("broken", {}).get("accuracy", 0)

    scribe_lift = scribe_acc - direct_acc
    broken_lift = broken_acc - direct_acc

    if scribe_lift <= 0.02:
        return "CoT provides minimal gain overall — baseline is strong."

    broken_fraction = broken_lift / scribe_lift if scribe_lift > 0 else 0

    if broken_fraction >= 0.75:
        return (
            f"OUTCOME B (statistical conditioning): broken traces recover "
            f"{broken_fraction:.0%} of scribe gains. CoT works via token-level "
            f"conditioning, not semantic content. This is the more surprising result."
        )
    elif broken_fraction <= 0.25:
        return (
            f"OUTCOME A (semantic content matters): broken traces recover only "
            f"{broken_fraction:.0%} of scribe gains. The model genuinely uses "
            f"intermediate reasoning steps."
        )
    else:
        return (
            f"MIXED: broken traces recover {broken_fraction:.0%} of scribe gains. "
            f"Both statistical conditioning and semantic content contribute."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="experiments/cot_mechanism/results")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to a specific model slug (e.g. gpt-4o-mini)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    slug = args.model.replace("/", "-").replace(".", "-") if args.model else None
    rows = load_results(results_dir, slug)

    if not rows:
        print("No results found.")
        return

    summary = summarize(rows)

    print("\n" + "=" * 60)
    print("  CoT Mechanism Experiment — Results")
    print("=" * 60)
    print(f"\n{'Condition':<12} {'Accuracy':>10} {'Correct':>8} {'N':>6} {'Avg trace chars':>16}")
    print("-" * 56)
    for cond in CONDITIONS:
        s = summary.get(cond)
        if not s:
            continue
        print(
            f"{cond:<12} {s['accuracy']:>10.1%} {s['correct']:>8} "
            f"{s['n']:>6} {s['avg_trace_chars']:>16.0f}"
        )

    print("\n" + "-" * 60)
    print("Interpretation:")
    print(interpret(summary))
    print("=" * 60 + "\n")

    # Per-problem breakdown: which problems flip between conditions?
    by_id_cond = defaultdict(dict)
    for r in rows:
        by_id_cond[r["id"]][r["condition"]] = r["correct"]

    flips = {"scribe_not_broken": 0, "broken_not_scribe": 0, "both": 0, "neither": 0}
    for pid, conds in by_id_cond.items():
        s = conds.get("scribe", False)
        b = conds.get("broken", False)
        if s and not b:
            flips["scribe_not_broken"] += 1
        elif b and not s:
            flips["broken_not_scribe"] += 1
        elif s and b:
            flips["both"] += 1
        else:
            flips["neither"] += 1

    n = len(by_id_cond)
    print("Per-problem breakdown (scribe vs. broken):")
    print(f"  Both correct:            {flips['both']:4d} ({flips['both']/n:.0%})")
    print(f"  Scribe only correct:     {flips['scribe_not_broken']:4d} ({flips['scribe_not_broken']/n:.0%})")
    print(f"  Broken only correct:     {flips['broken_not_scribe']:4d} ({flips['broken_not_scribe']/n:.0%})")
    print(f"  Neither correct:         {flips['neither']:4d} ({flips['neither']/n:.0%})")


if __name__ == "__main__":
    main()
