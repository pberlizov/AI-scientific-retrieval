"""Targeted follow-up sweep to strengthen the additivity-loss paper claim.

Two sub-experiments:
  1. Lambda sweep: fix S=5, d=20, k in {40, 80}; vary lambda over 5 values.
     Tests whether current lambda=0.1 is near-optimal or if there's headroom.
  2. d=40 sweep: fix S=5, lambda=0.1; run k=10..160 with d=40.
     Tests d-dependence of the additivity effect.

Usage:
    python3 targeted_sweep.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import train_model, evaluate_model

RESULTS_DIR = Path(__file__).parent / "results"
DEVICE = torch.device("cpu")
N_TRIALS = 5


def lambda_sweep():
    """Sub-experiment 1: lambda sensitivity at S=5, d=20, k=40 and k=80."""
    print("\n" + "="*60)
    print("Lambda sweep (S=5, d=20, k=[40,80], n_trials=5)")
    print("="*60)

    lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
    ks = [40, 80]
    results = []

    for k in ks:
        for lam in lambdas:
            trial_metrics = []
            for trial in range(N_TRIALS):
                model = train_model(
                    n_features=k, n_hidden=20, sparsity=5.0,
                    use_relu=True, condition="additivity",
                    lambda_reg=lam, n_steps=5000,
                    device=DEVICE, verbose=False,
                )
                m = evaluate_model(model, k, 5.0, DEVICE)
                trial_metrics.append(m)

            agg = {
                "experiment": "lambda_sweep",
                "k": k, "n_hidden": 20, "sparsity": 5.0, "lambda_reg": lam,
            }
            for key in trial_metrics[0]:
                vals = [m[key] for m in trial_metrics]
                agg[key] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))
            results.append(agg)

            print(f"  k={k:3d}  lambda={lam:.2f}: "
                  f"poly={agg['poly_index']:.3f}±{agg['poly_index_std']:.3f}  "
                  f"elhage={agg['elhage_feat']:.3f}  probe={agg['probe_acc']:.3f}")

    return results


def d40_sweep():
    """Sub-experiment 2: d=40, S=5, lambda=0.1, k=10..160."""
    print("\n" + "="*60)
    print("d=40 sweep (S=5, lambda=0.1, k=[10,20,40,60,80,120,160], n_trials=3)")
    print("="*60)

    ks = [10, 20, 40, 60, 80, 120, 160]
    results = []

    for condition in ["baseline", "additivity"]:
        for k in ks:
            trial_metrics = []
            for trial in range(3):
                model = train_model(
                    n_features=k, n_hidden=40, sparsity=5.0,
                    use_relu=True, condition=condition,
                    lambda_reg=0.1, n_steps=5000,
                    device=DEVICE, verbose=False,
                )
                m = evaluate_model(model, k, 5.0, DEVICE)
                trial_metrics.append(m)

            agg = {
                "experiment": "d40_sweep",
                "condition": condition,
                "k": k, "n_hidden": 40, "sparsity": 5.0, "lambda_reg": 0.1,
            }
            for key in trial_metrics[0]:
                vals = [m[key] for m in trial_metrics]
                agg[key] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))
            results.append(agg)

            print(f"  [{condition}] k={k:3d}: "
                  f"poly={agg['poly_index']:.3f}±{agg['poly_index_std']:.3f}  "
                  f"elhage={agg['elhage_feat']:.3f}  probe={agg['probe_acc']:.3f}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lambda_results = lambda_sweep()
    d40_results = d40_sweep()

    all_results = lambda_results + d40_results
    out = RESULTS_DIR / "targeted_sweep.jsonl"
    with open(out, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
