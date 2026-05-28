"""Compare SAE post-hoc decomposition vs. additivity-trained model.

For the baseline condition, the sweep already runs an SAE and records:
  sae_max_cosine_mean, sae_recovered_frac, sae_poly_sae_mean

This script loads those alongside the additivity condition's polysemanticity
and probe accuracy, and prints a head-to-head comparison table.

The key question: at k=80, d=20, S=5 —
  does an SAE applied post-hoc to a baseline model recover as many features
  as a model trained with the additivity regularizer?
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_sweep() -> dict:
    """Return nested dict: data[sparsity][k][condition] = record."""
    data = {}
    for f in sorted(RESULTS_DIR.glob("sweep_S*.jsonl")):
        for line in f.read_text().strip().splitlines():
            if line:
                r = json.loads(line)
                s = r["sparsity"]
                k = r["k"]
                c = r["condition"]
                data.setdefault(s, {}).setdefault(k, {})[c] = r
    return data


def main():
    data = load_sweep()

    for sparsity in sorted(data.keys()):
        has_sae = any(
            "sae_recovered_frac" in data[sparsity][k].get("baseline", {})
            for k in data[sparsity]
        )
        if not has_sae:
            continue

        print(f"\n{'='*70}")
        print(f"S={sparsity:.0f}  —  SAE (post-hoc on baseline) vs. Additivity training")
        print(f"{'='*70}")
        print(f"{'k':>5} | {'base poly':>10} {'add poly':>10} | "
              f"{'SAE cos':>8} {'SAE rec%':>9} {'SAE poly':>9} | "
              f"{'base probe':>11} {'add probe':>10}")
        print("-" * 85)

        for k in sorted(data[sparsity].keys()):
            base = data[sparsity][k].get("baseline", {})
            add  = data[sparsity][k].get("additivity", {})

            base_poly  = base.get("poly_index", float("nan"))
            add_poly   = add.get("poly_index",  float("nan"))
            sae_cos    = base.get("sae_max_cosine_mean", float("nan"))
            sae_rec    = base.get("sae_recovered_frac",  float("nan"))
            sae_poly   = base.get("sae_poly_sae_mean",   float("nan"))
            base_probe = base.get("probe_acc", float("nan"))
            add_probe  = add.get("probe_acc",  float("nan"))

            print(f"{k:>5} | {base_poly:>10.3f} {add_poly:>10.3f} | "
                  f"{sae_cos:>8.3f} {100*sae_rec:>8.1f}% {sae_poly:>9.3f} | "
                  f"{base_probe:>11.3f} {add_probe:>10.3f}")

    # Summary: at which k does additivity beat SAE on feature recovery?
    print("\n\nSummary: additivity poly vs. SAE recovered fraction (S=5)")
    s5 = data.get(5.0, {})
    print(f"{'k':>5}  {'add_poly':>10}  {'SAE_rec%':>10}  {'winner':>10}")
    for k in sorted(s5.keys()):
        base = s5[k].get("baseline", {})
        add  = s5[k].get("additivity", {})
        add_poly = add.get("poly_index", float("nan"))
        sae_rec  = base.get("sae_recovered_frac", float("nan"))
        # Lower poly and higher SAE recovery both mean "more monosemantic"
        # additivity wins if its poly is lower than (1 - SAE_rec)
        add_mono = 1.0 - add_poly          # fraction monosemantic under additivity
        winner = "additivity" if add_mono > sae_rec else "SAE" if sae_rec > add_mono + 0.05 else "tie"
        print(f"{k:>5}  {add_poly:>10.3f}  {100*sae_rec:>9.1f}%  {winner:>10}")


if __name__ == "__main__":
    main()
