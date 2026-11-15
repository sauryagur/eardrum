"""Acoustic encoder and the two CRNN heads for the ablation."""

from __future__ import annotations

import torch
from torch import nn

from eardrum import constants as C

ENCODER_DIM = 192
GRU_HIDDEN = 96
ENCODER_CHANNELS = 96


class AcousticEncoder(nn.Module):
    """2D CNN over (mel, time) that keeps frequency locality, then time-wise projection."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, ENCODER_CHANNELS, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(ENCODER_CHANNELS),
            nn.GELU(),
        )
        # Two stride-2 frequency halvings: 128 -> 64 -> 32.
        self.proj = nn.Linear(ENCODER_CHANNELS * (C.N_MELS // 4), ENCODER_DIM)
        self.dropout = nn.Dropout(0.15)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, 126) -> (B, 126, 192)."""
        hidden = self.conv(mel)
        batch, channels, freqs, frames = hidden.shape
        # (B, C, F, T) -> (B, T, C*F) with explicit permutes.
        hidden = hidden.permute(0, 3, 1, 2).reshape(batch, frames, channels * freqs)
        return self.dropout(self.proj(hidden))


class _BiGRU(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim, GRU_HIDDEN, batch_first=True, bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out  # (B, 126, 192)


class BaselineCRNN(nn.Module):
    """Single frame head; no onset signal anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AcousticEncoder()
        self.gru = _BiGRU(ENCODER_DIM)
        self.head = nn.Linear(2 * GRU_HIDDEN, C.N_PITCHES)

    def forward(self, mel: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.gru(self.encoder(mel))  # (B, 126, 192)
        frame_logits = self.head(hidden).permute(0, 2, 1)  # (B, 88, 126)
        return {"frame": frame_logits}


class OnsetsAndFramesCRNN(nn.Module):
    """Onset head plus a frame head conditioned on detached onset probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AcousticEncoder()
        self.onset_gru = _BiGRU(ENCODER_DIM)
        self.onset_head = nn.Linear(2 * GRU_HIDDEN, C.N_PITCHES)
        self.frame_gru = _BiGRU(ENCODER_DIM + C.N_PITCHES)
        self.frame_head = nn.Linear(2 * GRU_HIDDEN, C.N_PITCHES)

    def forward(self, mel: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(mel)  # (B, 126, 192)
        onset_hidden = self.onset_gru(encoded)
        onset_logits = self.onset_head(onset_hidden).permute(0, 2, 1)

        onset_probs = torch.sigmoid(onset_logits).detach().permute(0, 2, 1)
        frame_input = torch.cat([encoded, onset_probs], dim=-1)  # (B, 126, 280)
        frame_hidden = self.frame_gru(frame_input)
        frame_logits = self.frame_head(frame_hidden).permute(0, 2, 1)
        return {"onset": onset_logits, "frame": frame_logits}


MODELS = {
    "baseline": BaselineCRNN,
    "dual": OnsetsAndFramesCRNN,
}


def build_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
