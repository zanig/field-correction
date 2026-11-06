"""Perceiver-style case encoders and Fourier-lifted coordinate residual decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_TOKENS = 4
LATENT_DIM = 16
TOKEN_DIM = 32
NUM_HEADS = 4
FOURIER_OCTAVES = 6


class SetEncoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, TOKEN_DIM), nn.SiLU(),
            nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.SiLU(),
            nn.Linear(TOKEN_DIM, LATENT_DIM), nn.LayerNorm(LATENT_DIM),
        )
        self.queries = nn.Parameter(torch.randn(LATENT_TOKENS, LATENT_DIM) * 0.02)
        self.cross = nn.MultiheadAttention(LATENT_DIM, NUM_HEADS, batch_first=True)
        self.refine = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM, nhead=NUM_HEADS, dim_feedforward=TOKEN_DIM,
            activation="gelu", batch_first=True, norm_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        h = self.embed(points)
        q = self.queries.unsqueeze(0).expand(points.shape[0], -1, -1)
        attn, _ = self.cross(q, h, h, need_weights=False)
        return self.norm(self.refine(q + attn))


class Predictor(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, LATENT_DIM * 2), nn.SiLU(),
            nn.Linear(LATENT_DIM * 2, LATENT_DIM),
        )
        self.refine = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM, nhead=NUM_HEADS, dim_feedforward=TOKEN_DIM,
            activation="gelu", batch_first=True, norm_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(self.refine(z + self.pre(z)))


class Decoder(nn.Module):
    def __init__(self, query_dim: int, fourier_cols: tuple[int, ...]):
        super().__init__()
        self.fourier_cols = fourier_cols
        self.register_buffer(
            "bands", (2.0 ** torch.arange(FOURIER_OCTAVES, dtype=torch.float32)) * torch.pi
        )
        lifted = query_dim + len(fourier_cols) * 2 * FOURIER_OCTAVES

        self.project = nn.Sequential(
            nn.Linear(lifted, LATENT_DIM), nn.SiLU(), nn.LayerNorm(LATENT_DIM)
        )
        self.cross = nn.MultiheadAttention(LATENT_DIM, NUM_HEADS, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(LATENT_DIM + lifted, TOKEN_DIM), nn.SiLU(),
            nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.SiLU(),
            nn.LayerNorm(TOKEN_DIM),
            nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.SiLU(),
            nn.Linear(TOKEN_DIM, 3),
        )

    def lift(self, query: torch.Tensor) -> torch.Tensor:
        geom = query[..., list(self.fourier_cols)]
        phase = geom.unsqueeze(-1) * self.bands.view(1, 1, 1, -1)
        return torch.cat([query, torch.cat([torch.sin(phase), torch.cos(phase)], -1).flatten(-2)], -1)

    def forward(self, z: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        lifted = self.lift(query)
        q_proj = self.project(lifted)
        attended, _ = self.cross(q_proj, z, z, need_weights=False)
        return self.head(torch.cat([q_proj + attended, lifted], dim=-1))


def sigreg(z: torch.Tensor, slices: int = 128, points: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """Characteristic-function regularizer toward N(0, I) across batch/token rows."""
    x = z.contiguous().reshape(-1, z.shape[-1])
    if x.shape[0] < 2:
        return x.new_tensor(0.0)

    directions = F.normalize(torch.randn(x.shape[-1], slices, dtype=x.dtype, device=x.device), dim=0)
    projected = x @ directions

    t = torch.linspace(-t_max, t_max, points, dtype=x.dtype, device=x.device)
    t = t[t.abs() > 1e-6]
    phase = projected.unsqueeze(-1) * t.view(1, 1, -1)

    real = torch.cos(phase).mean(dim=0) - torch.exp(-0.5 * t.square()).view(1, -1)
    imag = torch.sin(phase).mean(dim=0)
    return real.square().mean() + imag.square().mean()


class Model(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.parts = nn.ModuleDict({
            # 10 channels: 4 geometry + 3 dimensionless residual (U_LES - U_RANS)/u_ref + 3 U_LES/u_ref
            "truth_encoder": SetEncoder(10, dropout), "truth_predictor": Predictor(dropout),
            "rans_encoder": SetEncoder(16, dropout), "rans_predictor": Predictor(dropout),
            "decoder": Decoder(16, (0, 1, 2, 14)),
        })

    def encode(self, context: torch.Tensor, truth: bool = False) -> torch.Tensor:
        kind = "truth" if truth else "rans"
        return self.parts[kind + "_predictor"](self.parts[kind + "_encoder"](context))

    def forward(self, latent: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.parts["decoder"](latent, query)
