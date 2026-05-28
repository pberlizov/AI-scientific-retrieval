"""Training script for the toy superposition model.

Supports three training conditions:
  baseline      — reconstruction loss only
  ortho         — + weight orthogonality regularizer (baseline comparison)
  additivity    — + activation-level additivity regularizer (our method)

Usage:
    python3 train.py [--conditions baseline ortho additivity] [--n-features 40 80 160]
                     [--n-hidden 20] [--sparsity 5] [--relu] [--steps 5000]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from toy_model import ToyModel, generate_batch, polysemanticity_index, elhage_polysemanticity
from additivity_loss import activation_additivity_loss, orthogonality_loss, weight_additivity_loss
from fano_bound import compute_per_feature_bounds, probe_accuracy
from sae import train_sae, sae_feature_recovery

RESULTS_DIR = Path(__file__).parent / "results"


def train_model(n_features: int, n_hidden: int, sparsity: float,
                use_relu: bool = True, condition: str = "baseline",
                lambda_reg: float = 0.1, n_steps: int = 5000,
                batch_size: int = 512, lr: float = 1e-3,
                device: torch.device = torch.device("cpu"),
                verbose: bool = True) -> ToyModel:

    model = ToyModel(n_features, n_hidden, use_relu=use_relu).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        x = generate_batch(batch_size, n_features, sparsity, device)
        x_hat, h = model(x)

        loss = model.reconstruction_loss(x, x_hat)

        if condition == "ortho":
            loss = loss + lambda_reg * orthogonality_loss(model)
        elif condition == "additivity":
            loss = loss + lambda_reg * activation_additivity_loss(model, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % 1000 == 0:
            print(f"  step {step+1:5d}  loss={loss.item():.4f}")

    model.eval()
    return model


def evaluate_model(model: ToyModel, n_features: int, sparsity: float,
                   device: torch.device, run_sae: bool = False) -> dict:
    p = 1.0 / sparsity
    bounds = compute_per_feature_bounds(model, p)
    acc = probe_accuracy(model, n_features, sparsity, device)
    poly = polysemanticity_index(model)
    elhage = elhage_polysemanticity(model)

    fano_unrecoverable = (bounds["pe_lb"] > 0.3).mean()

    result = {
        "probe_acc": acc,
        "poly_index": poly,
        "elhage_feat": elhage["poly_feature_mean"],
        "elhage_neur": elhage["poly_neuron_mean"],
        "fano_pe_mean": float(bounds["pe_lb"].mean()),
        "fano_pe_std": float(bounds["pe_lb"].std()),
        "fano_unrecoverable_frac": float(fano_unrecoverable),
        "gram_offdiag_rms": float(bounds["gram_off_rms"].mean()),
        "mi_lb_mean": float(bounds["mi_lb"].mean()),
    }

    if run_sae:
        sae = train_sae(model, n_features=n_features, sparsity=sparsity,
                        expansion=max(4.0, n_features / model.n_hidden),
                        device=device, n_steps=5000)
        sae_metrics = sae_feature_recovery(sae, model, n_features, sparsity, device)
        for k, v in sae_metrics.items():
            result[f"sae_{k}"] = v

    return result


def run_sweep(conditions: list[str], n_features_list: list[int], n_hidden: int,
              sparsity: float, use_relu: bool, n_steps: int, n_trials: int,
              lambda_reg: float, device: torch.device):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for condition in conditions:
        for k in n_features_list:
            print(f"\n[{condition}] k={k}, d={n_hidden}, S={sparsity:.1f}")
            trial_metrics = []

            for trial in range(n_trials):
                model = train_model(
                    n_features=k, n_hidden=n_hidden, sparsity=sparsity,
                    use_relu=use_relu, condition=condition,
                    lambda_reg=lambda_reg, n_steps=n_steps,
                    device=device, verbose=False,
                )
                run_sae = (condition == "baseline")  # SAE only on baseline for comparison
                metrics = evaluate_model(model, k, sparsity, device, run_sae=run_sae)
                trial_metrics.append(metrics)
                sae_str = f"  sae_rec={metrics.get('sae_recovered_frac', float('nan')):.2f}" if run_sae else ""
                print(f"  trial {trial+1}: probe_acc={metrics['probe_acc']:.3f}  "
                      f"poly={metrics['poly_index']:.3f}  "
                      f"elhage_feat={metrics['elhage_feat']:.3f}  "
                      f"fano_pe={metrics['fano_pe_mean']:.3f}{sae_str}")

            # Aggregate over trials
            agg = {
                "condition": condition,
                "k": k,
                "n_hidden": n_hidden,
                "sparsity": sparsity,
                "use_relu": use_relu,
            }
            for key in trial_metrics[0]:
                vals = [m[key] for m in trial_metrics]
                agg[key] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))
            all_results.append(agg)

    out_path = RESULTS_DIR / f"sweep_S{sparsity:.0f}.jsonl"
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults written to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "additivity"],
                        choices=["baseline", "ortho", "additivity"])
    parser.add_argument("--n-features", nargs="+", type=int,
                        default=[10, 20, 40, 60, 80, 120, 160])
    parser.add_argument("--n-hidden", type=int, default=20)
    parser.add_argument("--sparsity", nargs="+", type=float, default=[5.0],
                        help="One or more sparsity values to sweep over")
    parser.add_argument("--no-relu", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--out", default=None, help="Output JSONL path (default: results/sweep_results.jsonl)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    for s in args.sparsity:
        print(f"\n{'='*60}")
        print(f"Sparsity S={s}")
        print(f"Conditions: {args.conditions}, k={args.n_features}, d={args.n_hidden}")
        results = run_sweep(
            conditions=args.conditions,
            n_features_list=args.n_features,
            n_hidden=args.n_hidden,
            sparsity=s,
            use_relu=not args.no_relu,
            n_steps=args.steps,
            n_trials=args.trials,
            lambda_reg=args.lambda_reg,
            device=device,
        )
        all_results.extend(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        print(f"\nAll results written to {out_path}")


if __name__ == "__main__":
    main()
