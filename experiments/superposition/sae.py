"""Sparse Autoencoder (SAE) baseline for polysemanticity comparison.

Following Bricken et al. (2023) / Cunningham et al. (2023): train a sparse
autoencoder on the hidden representations of a trained toy model, then measure
how well the SAE's dictionary atoms align with original input features.

This gives a post-hoc comparison: does the additivity regularizer during training
achieve comparable or better monosemanticity than SAE-based decomposition?

The SAE is given an expansion factor E (default 4), so it has E*d dictionary
atoms to cover k input features. For a fair comparison we run it with E chosen
so that E*d ≥ k (enough atoms to represent all features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from toy_model import ToyModel, generate_batch


class SparseAutoencoder(nn.Module):
    def __init__(self, d_hidden: int, d_sae: int):
        super().__init__()
        self.encoder = nn.Linear(d_hidden, d_sae)
        self.decoder = nn.Linear(d_sae, d_hidden, bias=False)
        # Normalise decoder columns to unit norm (Bricken et al.)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, h: torch.Tensor):
        acts = F.relu(self.encoder(h))
        h_hat = self.decoder(acts)
        return h_hat, acts

    def normalise_decoder(self):
        """Keep decoder columns unit-norm throughout training."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )


def train_sae(
    toy_model: ToyModel,
    n_features: int,
    sparsity: float,
    expansion: float = 4.0,
    l1_coeff: float = 1e-3,
    n_steps: int = 10000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> SparseAutoencoder:
    d = toy_model.n_hidden
    d_sae = max(int(expansion * d), n_features)

    sae = SparseAutoencoder(d, d_sae).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    toy_model.eval()
    for step in range(n_steps):
        with torch.no_grad():
            x = generate_batch(batch_size, n_features, sparsity, device)
            _, h = toy_model(x)

        h_hat, acts = sae(h)
        recon_loss = F.mse_loss(h_hat, h)
        l1_loss = acts.abs().mean()
        loss = recon_loss + l1_coeff * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalise_decoder()

    return sae


def sae_feature_recovery(
    sae: SparseAutoencoder,
    toy_model: ToyModel,
    n_features: int,
    sparsity: float,
    device: torch.device,
    n_samples: int = 4000,
) -> dict:
    """
    Measure how well SAE atoms align with the toy model's ground-truth features.

    For each input feature i, find the SAE atom that most closely matches
    the toy model's weight vector w_i (column i of W).

    Metrics:
      max_cosine_mean  : mean of the max cosine similarity across features
                         (1.0 = every feature has a dedicated SAE atom)
      recovered_frac   : fraction of features with max cosine ≥ 0.9
      poly_sae_mean    : mean Elhage polysemanticity of SAE decoder columns
    """
    toy_model.eval()
    sae.eval()

    W = toy_model.W.detach()                       # [d, k]
    W_hat = F.normalize(W, dim=0)                  # unit-norm columns
    D = sae.decoder.weight.detach()                # [d, d_sae] (unit-norm cols)

    # Cosine similarity: [k, d_sae]
    cosine = (W_hat.T @ D).abs()
    max_cosine = cosine.max(dim=1).values          # [k]

    # Elhage polysemanticity of SAE decoder atoms
    D2 = D ** 2
    col_sum2 = D2.sum(dim=0).clamp(min=1e-12)
    col_sum4 = (D ** 4).sum(dim=0)
    poly_sae = (1.0 - col_sum4 / col_sum2 ** 2).mean().item()

    return {
        "max_cosine_mean": max_cosine.mean().item(),
        "max_cosine_std": max_cosine.std().item(),
        "recovered_frac": (max_cosine >= 0.9).float().mean().item(),
        "poly_sae_mean": poly_sae,
        "d_sae": sae.decoder.weight.shape[1],
    }
