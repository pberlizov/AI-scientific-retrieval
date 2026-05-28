"""Analyze sweep results and print the paper's tables.

Usage:
    python3 analyze.py
    python3 analyze.py --results results/sweep_results.jsonl
"""

import argparse
import json
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def print_table(results: list[dict], metric: str, label: str):
    conditions = sorted(set(r["condition"] for r in results))
    ks = sorted(set(r["k"] for r in results))
    n_hidden = results[0]["n_hidden"]

    print(f"\n{label} (d={n_hidden})")
    print("-" * (12 + 12 * len(conditions)))
    header = f"{'k':>6}  {'d/k':>5}  " + "  ".join(f"{c:>12}" for c in conditions)
    print(header)
    print("-" * len(header))

    for k in ks:
        row = f"{k:>6}  {n_hidden/k:>5.2f}  "
        for cond in conditions:
            match = [r for r in results if r["k"] == k and r["condition"] == cond]
            if match:
                val = match[0][metric]
                std = match[0].get(metric + "_std", 0.0)
                row += f"  {val:6.3f}±{std:.3f}"
            else:
                row += f"  {'—':>12}"
        print(row)


def print_fano_vs_probe(results: list[dict]):
    """Key table: does the Fano bound predict probe accuracy?"""
    baseline = [r for r in results if r["condition"] == "baseline"]
    if not baseline:
        return

    print("\n=== FANO BOUND vs EMPIRICAL PROBE ACCURACY (baseline) ===")
    print(f"{'k':>6}  {'d/k':>5}  {'Fano P_e≥':>10}  {'Probe acc':>10}  {'Gap':>8}  {'Poly':>8}")
    print("-" * 60)
    for r in sorted(baseline, key=lambda x: x["k"]):
        pe = r["fano_pe_mean"]
        acc = r["probe_acc"]
        poly = r["poly_index"]
        gap = acc - (1 - pe)   # positive = model exceeds Fano lower bound (expected)
        print(f"{r['k']:>6}  {r['n_hidden']/r['k']:>5.2f}  "
              f"{pe:>10.3f}  {acc:>10.3f}  {gap:>+8.3f}  {poly:>8.3f}")

    print("\nNote: Fano P_e is a LOWER BOUND on error. "
          "Acc should be ≤ 1−Fano_Pe is never required; "
          "when Fano_Pe→0.5 the feature is predicted unrecoverable.")


def print_regularizer_comparison(results: list[dict]):
    print("\n=== REGULARIZER COMPARISON: poly index vs probe accuracy ===")
    conditions = sorted(set(r["condition"] for r in results))
    ks = sorted(set(r["k"] for r in results))

    print(f"\n{'':20}" + "  ".join(f"{'k='+str(k):>18}" for k in ks))
    for cond in conditions:
        row = f"{cond:20}"
        for k in ks:
            match = [r for r in results if r["k"] == k and r["condition"] == cond]
            if match:
                r = match[0]
                row += f"  acc={r['probe_acc']:.2f} poly={r['poly_index']:.2f}"
            else:
                row += f"  {'—':>18}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/superposition/results/sweep_results.jsonl")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"Results file not found: {path}")
        print("Run train.py first.")
        return

    results = load_results(path)
    print(f"Loaded {len(results)} result rows.")

    print_fano_vs_probe(results)
    print_table(results, "probe_acc", "Probe Accuracy")
    print_table(results, "poly_index", "Polysemanticity Index")
    print_table(results, "fano_pe_mean", "Mean Fano P_e lower bound")
    print_regularizer_comparison(results)


if __name__ == "__main__":
    main()
