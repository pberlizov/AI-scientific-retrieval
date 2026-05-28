"""Sigma-algebra additivity regularizer (Entry L).

THEORY
------
Define the feature measure  μ_W(A) = ||Σ_{i∈A} w_i||²  for any subset A of features.

μ_W is additive over disjoint sets iff all cross-set inner products vanish:
  μ_W(A∪B) = μ_W(A) + μ_W(B)  ⟺  ⟨Σ_{i∈A} w_i, Σ_{j∈B} w_j⟩ = 0  ∀ disjoint A,B

For LINEAR models this collapses to pairwise orthogonality of weight vectors.

For RELU models the analogue operates on the OUTPUT of the encoder:
  Let φ(x) = ReLU(Wx + b). Then for a random partition (x_A, x_B) of x:
    L_add = E[ ||φ(x_A) + φ(x_B) − φ(x)||² ]
  which is zero iff ReLU introduces no "cancellation" between the two parts.
  This is STRICTLY STRONGER than weight-level orthogonality: pairwise orthogonal
  weight vectors can still produce non-additive hidden representations when ReLU
  creates sign-dependent interference.

Both forms are implemented below. Experiments use the ReLU form (more expressive)
and compare against the weight-level orthogonality baseline.
"""

import torch
import torch.nn.functional as F
from toy_model import ToyModel


# ─── Weight-level additivity (linear analogue) ────────────────────────────────

def weight_additivity_loss(model: ToyModel, n_samples: int = 64) -> torch.Tensor:
    """
    Penalise Σ_{A,B disjoint} ⟨Σ_{i∈A} w_i, Σ_{j∈B} w_j⟩²

    Estimated by sampling random binary partitions of the feature index set.
    For each sample: draw a random mask m ∈ {0,1}^k; A = {i: m_i=1}, B = complement.
    """
    W = model.W  # [n_hidden, k]
    k = W.shape[1]
    device = W.device

    masks = torch.bernoulli(
        torch.full((n_samples, k), 0.5, device=device)
    )  # [n_samples, k]

    # Σ_{i∈A} w_i  and  Σ_{j∈B} w_j
    sA = masks @ W.T          # [n_samples, n_hidden]
    sB = (1 - masks) @ W.T    # [n_samples, n_hidden]

    dot = (sA * sB).sum(dim=1)   # [n_samples]
    return (dot ** 2).mean()


def orthogonality_loss(model: ToyModel) -> torch.Tensor:
    """Baseline: penalise all pairwise off-diagonal Gram entries squared."""
    W = model.W  # [n_hidden, k]
    G = W.T @ W  # [k, k]
    k = G.shape[0]
    off_diag = G - torch.diag(torch.diag(G))
    return (off_diag ** 2).sum() / (k * (k - 1))


# ─── Activation-level additivity (ReLU analogue) ─────────────────────────────

def activation_additivity_loss(model: ToyModel, x: torch.Tensor) -> torch.Tensor:
    """
    For ReLU models: penalise how much the encoder fails to additively decompose
    over random partitions of active features.

    L_add = E_split[ ||encode(x_A) + encode(x_B) − encode(x)||² ]
              / (||encode(x)||² + ε)    (normalised)

    The normalisation ensures the loss is scale-invariant and doesn't collapse
    to zero simply by shrinking W.
    """
    device = x.device
    batch, k = x.shape

    # Random binary split of ACTIVE features only
    active = (x > 0).float()
    split = torch.bernoulli(0.5 * active)  # 1 → goes to A, 0 → goes to B
    x_A = x * split
    x_B = x * (active - split)

    h_full = model.encode(x)
    h_A = model.encode(x_A)
    h_B = model.encode(x_B)

    residual = h_A + h_B - h_full
    denom = (h_full.detach() ** 2).mean() + 1e-8
    return (residual ** 2).mean() / denom
