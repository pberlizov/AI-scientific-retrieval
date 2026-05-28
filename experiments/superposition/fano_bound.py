"""Fano-bound prediction of the superposition phase transition.

For each feature i in a trained model, we compute:

  1. The effective SNR for recovering X_i from the representation, using the
     Gram matrix of weight vectors and the Gaussian interference approximation.

  2. A lower bound on the per-feature probing error via the data-processing
     inequality and Fano's inequality:
       P_e(X_i) >= h_b^{-1}( h_b(p) - I_lb(X_i; h) )
     where h_b is binary entropy, p is feature prior, and I_lb is the
     Gaussian-channel mutual information lower bound.

  3. The predicted phase-transition threshold k* = largest k such that the
     mean Fano bound is below a target error rate ε.

Key derivation:
  The Gaussian channel approximation treats the interference from other
  active features as additive Gaussian noise:
    σ²_i = p(1−p) · Σ_{j≠i} (w_i^T w_j)²
  giving  I_lb(X_i; h) = ½ log(1 + p(1−p) ||w_i||⁴ / σ²_i).
"""

import math
import torch
import numpy as np
from toy_model import ToyModel


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def inverse_binary_entropy(h: float) -> float:
    """Smallest p in [0, 0.5] such that h_b(p) = h (bits). Returns 0 if h <= 0."""
    if h <= 0:
        return 0.0
    if h >= 1.0:
        return 0.5
    # Binary search on [0, 0.5]
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if binary_entropy(mid) < h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_per_feature_bounds(model: ToyModel, p: float) -> dict:
    """
    Returns per-feature Fano lower bounds on probing error and related quantities.

    p : feature prior probability (= 1/sparsity)
    """
    W = model.W.detach().cpu()     # [n_hidden, n_features]
    k = W.shape[1]
    G = W.T @ W                    # [k, k] Gram matrix

    h_prior = binary_entropy(p)    # bits

    snrs, mi_lbs, pe_lbs, h_conds = [], [], [], []

    # Normalize weight vectors to unit norm so SNR measures geometry, not scale.
    # Scale only affects how well the model uses each direction; normalization
    # gives a pure geometric bound independent of weight magnitude.
    col_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_hat = W / col_norms          # [n_hidden, k], unit-norm columns
    G_hat = W_hat.T @ W_hat        # [k, k] cosine similarity Gram matrix

    for i in range(k):
        # With unit-norm vectors, G_hat[i,i] = 1 always.
        # Interference = sum of squared cosine similarities with other features.
        cross = G_hat[i].clone()
        cross[i] = 0.0
        sigma_sq = (p * (1 - p) * cross ** 2).sum().item()

        # Gaussian-channel MI lower bound.
        # SNR = signal power / interference variance.
        # With unit-norm w_i: signal power = p(1-p) * ||w_i||^4 → p(1-p) * 1.
        if sigma_sq < 1e-12:
            mi_lb = h_prior
        else:
            snr = p * (1 - p) / sigma_sq
            mi_lb = 0.5 * math.log2(1 + snr)

        snr_val = p * (1 - p) / (sigma_sq + 1e-12)
        h_cond = max(0.0, h_prior - mi_lb)
        pe_lb = inverse_binary_entropy(h_cond)

        snrs.append(snr_val)
        mi_lbs.append(mi_lb)
        h_conds.append(h_cond)
        pe_lbs.append(pe_lb)

    return {
        "snr": np.array(snrs),
        "mi_lb": np.array(mi_lbs),        # MI lower bound (bits)
        "h_cond": np.array(h_conds),       # residual conditional entropy
        "pe_lb": np.array(pe_lbs),         # Fano lower bound on P_e
        "gram_diag": np.array([G[i, i].item() for i in range(k)]),
        "gram_off_rms": np.array([
            math.sqrt(max(0.0, (G_hat[i] ** 2 - G_hat[i, i] ** 2).sum().item()) / max(k - 1, 1))
            for i in range(k)
        ]),
    }


def predict_phase_transition(n_hidden: int, n_features_range: range,
                              sparsity: float, n_trials: int = 5,
                              use_relu: bool = True,
                              device: torch.device = torch.device("cpu")) -> dict:
    """
    Sweep k (number of features) and return, for each k:
      - mean Fano bound on P_e
      - mean empirical probing accuracy
      - polysemanticity index

    Trains a fresh model for each k, each averaged over n_trials.
    """
    from train import train_model
    import numpy as np

    results = {"k": [], "pe_lb_mean": [], "probe_acc_mean": [], "poly_mean": []}

    for k in n_features_range:
        pe_lbs_trials, accs_trials, polys_trials = [], [], []
        for _ in range(n_trials):
            model = train_model(n_features=k, n_hidden=n_hidden,
                                sparsity=sparsity, use_relu=use_relu,
                                n_steps=3000, device=device, verbose=False)
            bounds = compute_per_feature_bounds(model, p=1.0 / sparsity)
            pe_lbs_trials.append(bounds["pe_lb"].mean())

            acc = probe_accuracy(model, k, sparsity, device)
            accs_trials.append(acc)

            from toy_model import polysemanticity_index
            polys_trials.append(polysemanticity_index(model))

        results["k"].append(k)
        results["pe_lb_mean"].append(np.mean(pe_lbs_trials))
        results["probe_acc_mean"].append(np.mean(accs_trials))
        results["poly_mean"].append(np.mean(polys_trials))
        print(f"  k={k:3d}  Fano P_e≥{results['pe_lb_mean'][-1]:.3f}  "
              f"probe_acc={results['probe_acc_mean'][-1]:.3f}  "
              f"poly={results['poly_mean'][-1]:.3f}")

    return results


def probe_accuracy(model: ToyModel, n_features: int, sparsity: float,
                   device: torch.device, n_samples: int = 2000) -> float:
    """Train a linear probe per feature; return mean accuracy."""
    from toy_model import generate_batch
    model.eval()

    with torch.no_grad():
        x = generate_batch(n_samples, n_features, sparsity, device)
        _, h = model(x)

    h_np = h.cpu().numpy()
    x_np = x.cpu().numpy()

    from sklearn.linear_model import LogisticRegression
    accs = []
    for i in range(n_features):
        y = (x_np[:, i] > 0).astype(int)
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(h_np, y)
        accs.append(clf.score(h_np, y))
    return float(np.mean(accs)) if accs else 0.0
