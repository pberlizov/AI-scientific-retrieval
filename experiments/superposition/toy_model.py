"""Elhage et al. (2022) toy superposition model.

A minimal model that exhibits superposition: k features mapped to d < k
dimensions, with sparse inputs. The canonical testbed for superposition research.

Two variants:
  linear: h = Wx,        x̂ = W^T h         (W ∈ R^{d×k})
  relu:   h = ReLU(Wx+b), x̂ = W^T h + b'   (same W for encoder and decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, use_relu: bool = True,
                 feature_importance: torch.Tensor | None = None):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.use_relu = use_relu

        self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_hidden))
        self.b_out = nn.Parameter(torch.zeros(n_features))

        # Feature importance weights (default: uniform)
        if feature_importance is None:
            feature_importance = torch.ones(n_features)
        self.register_buffer("importance", feature_importance)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W.T + self.b
        if self.use_relu:
            h = F.relu(h)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W + self.b_out

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Importance-weighted MSE (Elhage et al. eq. 1)."""
        return (self.importance * (x - x_hat) ** 2).mean()


def generate_batch(batch_size: int, n_features: int, sparsity: float,
                   device: torch.device) -> torch.Tensor:
    """Each feature independently active with probability 1/sparsity, uniform value in [0,1]."""
    prob = 1.0 / sparsity
    mask = torch.bernoulli(torch.full((batch_size, n_features), prob, device=device))
    values = torch.rand(batch_size, n_features, device=device)
    return mask * values


def polysemanticity_index(model: ToyModel) -> float:
    """
    Feature-level: fraction of features with no dominant hidden dimension.

    A feature i is monosemantic if its weight vector W[:,i] is nearly aligned
    with a single hidden basis direction — max(|W[:,i]|) / ||W[:,i]|| ≈ 1.
    Returns fraction of features where this ratio < 0.9.
    """
    W = model.W.detach()  # [n_hidden, n_features]
    norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W / norms
    max_alignment = W_norm.abs().max(dim=0).values
    return (max_alignment < 0.9).float().mean().item()


def elhage_polysemanticity(model: ToyModel) -> dict:
    """
    Elhage et al. (2022) polysemanticity metrics.

    Feature polysemanticity (how distributed is feature i across neurons):
      poly_feature(i) = 1 - Σ_j W_ji^4 / (Σ_j W_ji^2)^2
      = 0 if feature i activates exactly one neuron (monosemantic)
      = 1 - 1/d if all neurons equally represent feature i

    Neuron polysemanticity (how many features does neuron j represent):
      poly_neuron(j) = 1 - Σ_i W_ji^4 / (Σ_i W_ji^2)^2
      = 0 if neuron j is activated by exactly one feature

    Returns mean over features and over neurons separately.
    """
    W = model.W.detach()  # [n_hidden, n_features]

    # Feature polysemanticity (column-wise)
    W2_col = W ** 2                          # [n_hidden, n_features]
    col_sum2 = W2_col.sum(dim=0).clamp(min=1e-12)   # [n_features]
    col_sum4 = (W ** 4).sum(dim=0)
    poly_feature = 1.0 - col_sum4 / (col_sum2 ** 2)  # [n_features]

    # Neuron polysemanticity (row-wise)
    row_sum2 = W2_col.sum(dim=1).clamp(min=1e-12)    # [n_hidden]
    row_sum4 = (W ** 4).sum(dim=1)
    poly_neuron = 1.0 - row_sum4 / (row_sum2 ** 2)   # [n_hidden]

    return {
        "poly_feature_mean": poly_feature.mean().item(),
        "poly_feature_std": poly_feature.std().item(),
        "poly_neuron_mean": poly_neuron.mean().item(),
        "poly_neuron_std": poly_neuron.std().item(),
    }
