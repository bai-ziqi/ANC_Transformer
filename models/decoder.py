"""Eight bank-derived queries attend to noise memory (Q=bank; K/V=noise)."""
from __future__ import annotations

import torch
from torch import nn


class WienerDecoder(nn.Module):
    def __init__(self, taps, scene_count, cfg):
        super().__init__()
        width = cfg["d_model"]
        self.projection = nn.Linear(taps, width)
        self.scene_embedding = nn.Embedding(scene_count, width)
        layer = nn.TransformerDecoderLayer(width, cfg["nhead"], cfg["dim_feedforward"],
                                           cfg["dropout"], batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg["decoder_layers"], norm=nn.LayerNorm(width))

    def forward(self, bank, memory):
        identities = torch.arange(len(bank), device=bank.device)
        # One shared scalar rescales projection inputs without discarding
        # relative FIR amplitudes. Actual FIR coefficients stay unchanged.
        scale = bank.square().mean().sqrt().clamp_min(1e-8)
        tokens = self.projection(bank / scale) + self.scene_embedding(identities)
        queries = tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
        # These 8 tokens are a candidate set, not an autoregressive sequence.
        # No triangular query mask; decoder self-attention is ordinary set attention.
        return self.decoder(tgt=queries, memory=memory)
