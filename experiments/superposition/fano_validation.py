"""Validate that Fano P_e bounds predict the empirical probe accuracy drop.

Loads the sparsity sweep results and for each (condition, k, S) triple:
  - extracts the Fano P_e lower bound (fano_pe_mean)
  - extracts empirical probe accuracy
  - checks whether the bound is tight (P_e ≈ 1 - probe_acc)

Prints a table and saves a summary JSON.
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_sweep() -> list[dict]:
    records = []
    for f in sorted(RESULTS_DIR.glob("sweep_S*.jsonl")):
        for line in f.read_text().strip().splitlines():
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_sweep()

    print(f"{'S':>4} {'k':>4} {'cond':>12} {'probe_acc':>10} {'emp_pe':>8} "
          f"{'fano_pe':>8} {'ratio':>7} {'tight?':>7}")
    print("-" * 65)

    summary = []
    for r in records:
        emp_pe = 1.0 - r["probe_acc"]        # empirical error rate
        fano_pe = r["fano_pe_mean"]           # theoretical lower bound
        # ratio: how close is the bound to the empirical error?
        # ratio > 1 would violate Fano (shouldn't happen if bound is correct)
        # ratio close to 1 means the bound is tight
        ratio = fano_pe / emp_pe if emp_pe > 1e-4 else float("nan")
        tight = "YES" if (0.5 <= ratio <= 1.05 if not np.isnan(ratio) else False) else "-"

        print(f"{r['sparsity']:>4.0f} {r['k']:>4d} {r['condition']:>12s} "
              f"{r['probe_acc']:>10.3f} {emp_pe:>8.3f} {fano_pe:>8.4f} "
              f"{ratio:>7.3f} {tight:>7}")

        summary.append({
            "sparsity": r["sparsity"], "k": r["k"], "condition": r["condition"],
            "probe_acc": r["probe_acc"], "emp_pe": emp_pe,
            "fano_pe_mean": fano_pe, "fano_pe_std": r.get("fano_pe_std", 0),
            "ratio": ratio if not np.isnan(ratio) else None,
        })

    out = RESULTS_DIR / "fano_validation.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}")

    # Summary stats
    valid = [s for s in summary if s["ratio"] is not None and s["emp_pe"] > 0.01]
    ratios = [s["ratio"] for s in valid]
    violations = [s for s in valid if s["ratio"] > 1.05]
    tight = [s for s in valid if 0.5 <= s["ratio"] <= 1.05]

    print(f"\nFano bound summary ({len(valid)} cases with emp_pe > 1%):")
    print(f"  Mean ratio (fano_pe / emp_pe): {np.mean(ratios):.3f}")
    print(f"  Violations (ratio > 1.05):     {len(violations)}")
    print(f"  Tight (ratio 0.5–1.05):        {len(tight)}/{len(valid)}")

    # Phase transition: where does probe_acc drop and does fano_pe predict it?
    print("\nPhase transition check (S=5, baseline):")
    s5_base = sorted(
        [r for r in records if r["sparsity"] == 5.0 and r["condition"] == "baseline"],
        key=lambda r: r["k"]
    )
    for r in s5_base:
        emp_pe = 1.0 - r["probe_acc"]
        fano_pe = r["fano_pe_mean"]
        print(f"  k={r['k']:3d}: probe_acc={r['probe_acc']:.3f}  "
              f"emp_pe={emp_pe:.3f}  fano_pe={fano_pe:.4f}  "
              f"gram_rms={r.get('gram_offdiag_rms', float('nan')):.4f}")


if __name__ == "__main__":
    main()
