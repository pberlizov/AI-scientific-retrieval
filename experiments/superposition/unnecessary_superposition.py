"""Investigate 'unnecessary superposition': why does gradient descent find
distributed representations even when k ≤ d (more dimensions than features)?

Three hypotheses:
  H1 (Symmetry): The loss landscape has infinitely many perfect-reconstruction
     solutions when k < d; gradient descent finds a distributed one.
  H2 (Initialization): Kaiming random init breaks the symmetry toward
     distributed solutions; orthogonal init would find monosemantic ones.
  H3 (Training time): More steps would converge to monosemantic solutions.

Tests:
  1. Consistency: train 20 baseline models at k=10, d=20; plot poly distribution.
  2. Init experiment: compare kaiming vs orthogonal-column init.
  3. Steps experiment: train to 50k steps; does poly decrease?
  4. Monosemantic initialization: start from a monosemantic W; verify it's a
     local minimum (poly stays low after training).

Usage:
    python3 unnecessary_superposition.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from toy_model import ToyModel, generate_batch, polysemanticity_index, elhage_polysemanticity
from train import train_model

RESULTS_DIR = Path(__file__).parent / "results"


def monosemantic_init(model: ToyModel):
    """
    Initialise W so each feature maps to a dedicated hidden dimension.
    For k ≤ d: assign feature i to hidden dimension i, zero elsewhere.
    This is a monosemantic solution by construction.
    """
    k = model.n_features
    d = model.n_hidden
    with torch.no_grad():
        model.W.data.zero_()
        for i in range(min(k, d)):
            model.W.data[i, i] = 1.0


def orthogonal_column_init(model: ToyModel):
    """
    Initialise W so columns are random but orthonormal (up to k ≤ d constraint).
    Uses the first k columns of a random d×d orthogonal matrix.
    """
    k = model.n_features
    d = model.n_hidden
    with torch.no_grad():
        Q = torch.linalg.qr(torch.randn(d, d))[0]
        model.W.data = Q[:, :k]


def run_consistency_test(n_trials: int = 20, n_features: int = 10,
                         n_hidden: int = 20, sparsity: float = 5.0,
                         n_steps: int = 5000, device: torch.device = torch.device("cpu")) -> dict:
    """H1/H2: is unnecessary superposition consistent across random seeds?"""
    print(f"\n=== Consistency test (k={n_features}, d={n_hidden}, {n_trials} trials) ===")
    poly_vals, elhage_feat_vals, elhage_neur_vals = [], [], []

    for t in range(n_trials):
        model = train_model(n_features=n_features, n_hidden=n_hidden,
                            sparsity=sparsity, condition="baseline",
                            n_steps=n_steps, device=device, verbose=False)
        p = polysemanticity_index(model)
        e = elhage_polysemanticity(model)
        poly_vals.append(p)
        elhage_feat_vals.append(e["poly_feature_mean"])
        elhage_neur_vals.append(e["poly_neuron_mean"])
        print(f"  trial {t+1:2d}: poly={p:.3f}  elhage_feat={e['poly_feature_mean']:.3f}  "
              f"elhage_neur={e['poly_neuron_mean']:.3f}")

    return {
        "test": "consistency",
        "n_features": n_features, "n_hidden": n_hidden, "sparsity": sparsity,
        "poly_mean": float(np.mean(poly_vals)), "poly_std": float(np.std(poly_vals)),
        "elhage_feat_mean": float(np.mean(elhage_feat_vals)),
        "elhage_neur_mean": float(np.mean(elhage_neur_vals)),
    }


def run_init_experiment(n_features: int = 10, n_hidden: int = 20,
                        sparsity: float = 5.0, n_steps: int = 5000,
                        n_trials: int = 5, device: torch.device = torch.device("cpu")) -> list[dict]:
    """H2: does init determine final polysemanticity?"""
    print(f"\n=== Init experiment (k={n_features}, d={n_hidden}) ===")
    results = []

    for init_name, init_fn in [
        ("kaiming", None),
        ("orthogonal_cols", orthogonal_column_init),
        ("monosemantic", monosemantic_init),
    ]:
        polys, recons = [], []
        for t in range(n_trials):
            model = ToyModel(n_features, n_hidden, use_relu=True).to(device)
            if init_fn is not None:
                init_fn(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(n_steps):
                x = generate_batch(512, n_features, sparsity, device)
                x_hat, _ = model(x)
                loss = model.reconstruction_loss(x, x_hat)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                x_test = generate_batch(1000, n_features, sparsity, device)
                x_hat_test, _ = model(x_test)
                recon = model.reconstruction_loss(x_test, x_hat_test).item()
            poly = polysemanticity_index(model)
            polys.append(poly); recons.append(recon)

        row = {"init": init_name, "poly_mean": float(np.mean(polys)),
               "poly_std": float(np.std(polys)), "recon_mean": float(np.mean(recons))}
        results.append(row)
        print(f"  {init_name:20s}: poly={row['poly_mean']:.3f}±{row['poly_std']:.3f}  "
              f"recon={row['recon_mean']:.5f}")

    return results


def run_steps_experiment(n_features: int = 10, n_hidden: int = 20,
                         sparsity: float = 5.0,
                         step_checkpoints: list[int] = None,
                         device: torch.device = torch.device("cpu")) -> list[dict]:
    """H3: does more training time reduce unnecessary superposition?"""
    if step_checkpoints is None:
        step_checkpoints = [1000, 5000, 20000, 50000]

    print(f"\n=== Steps experiment (k={n_features}, d={n_hidden}) ===")
    model = ToyModel(n_features, n_hidden, use_relu=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = []
    step = 0
    for target in step_checkpoints:
        while step < target:
            x = generate_batch(512, n_features, sparsity, device)
            x_hat, _ = model(x)
            loss = model.reconstruction_loss(x, x_hat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            step += 1

        model.eval()
        poly = polysemanticity_index(model)
        e = elhage_polysemanticity(model)
        with torch.no_grad():
            x_test = generate_batch(1000, n_features, sparsity, device)
            x_hat_test, _ = model(x_test)
            recon = model.reconstruction_loss(x_test, x_hat_test).item()
        model.train()

        row = {"steps": step, "poly": poly, "elhage_feat": e["poly_feature_mean"], "recon": recon}
        results.append(row)
        print(f"  step {step:6d}: poly={poly:.3f}  elhage_feat={e['poly_feature_mean']:.3f}  "
              f"recon={recon:.5f}")

    return results


def run_landscape_test(n_features: int = 10, n_hidden: int = 20,
                       sparsity: float = 5.0, n_steps: int = 5000,
                       device: torch.device = torch.device("cpu")) -> dict:
    """
    Is the monosemantic solution actually a local minimum?
    Start from monosemantic init, train with baseline loss, measure if poly stays low.
    """
    print(f"\n=== Landscape test: is monosemantic a local minimum? (k={n_features}, d={n_hidden}) ===")

    model = ToyModel(n_features, n_hidden, use_relu=True).to(device)
    monosemantic_init(model)

    poly_before = polysemanticity_index(model)
    e_before = elhage_polysemanticity(model)
    print(f"  Before training: poly={poly_before:.3f}  elhage_feat={e_before['poly_feature_mean']:.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_steps):
        x = generate_batch(512, n_features, sparsity, device)
        x_hat, _ = model(x)
        loss = model.reconstruction_loss(x, x_hat)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    poly_after = polysemanticity_index(model)
    e_after = elhage_polysemanticity(model)
    with torch.no_grad():
        x_test = generate_batch(1000, n_features, sparsity, device)
        x_hat_test, _ = model(x_test)
        recon = model.reconstruction_loss(x_test, x_hat_test).item()
    print(f"  After  training: poly={poly_after:.3f}  elhage_feat={e_after['poly_feature_mean']:.3f}  "
          f"recon={recon:.5f}")

    return {
        "test": "landscape",
        "poly_before": poly_before, "poly_after": poly_after,
        "elhage_feat_before": e_before["poly_feature_mean"],
        "elhage_feat_after": e_after["poly_feature_mean"],
        "recon": recon,
        "stayed_monosemantic": poly_after < 0.1,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    all_results = {}

    all_results["consistency"] = run_consistency_test(device=device)
    all_results["init"] = run_init_experiment(device=device)
    all_results["steps"] = run_steps_experiment(device=device)
    all_results["landscape"] = run_landscape_test(device=device)

    out = RESULTS_DIR / "unnecessary_superposition.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out}")

    # Summary
    print("\n=== SUMMARY ===")
    c = all_results["consistency"]
    print(f"Consistency: poly={c['poly_mean']:.3f}±{c['poly_std']:.3f} across 20 trials")
    print(f"  → {'CONSISTENT' if c['poly_std'] < 0.05 else 'VARIABLE'} (H1: symmetry degeneracy)")

    for r in all_results["init"]:
        print(f"Init {r['init']:20s}: poly={r['poly_mean']:.3f}±{r['poly_std']:.3f}")

    lm = all_results["landscape"]
    print(f"Landscape: monosemantic init {'stays monosemantic' if lm['stayed_monosemantic'] else 'drifts polysemantic'} after training")
    print(f"  → {'Monosemantic IS a local min' if lm['stayed_monosemantic'] else 'Monosemantic is NOT stable'}")


if __name__ == "__main__":
    main()
