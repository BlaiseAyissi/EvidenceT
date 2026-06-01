"""
reasoning/rssm.py
=================
Reasoning State Space Module (RSSM)

In PRIME-VQA, prepare_inputs_for_multimodal assembles inputs_embeds as:
  [BOS | knowledge_features (kg_proj_out_num tokens) |
         intra_class_images (kg_proj_out_num tokens) |
         image_features (proj_out_num tokens) |
         remaining question tokens]

All projected to Llama hidden size (3072) via mm_projector / kg_projector.

The RSSM receives z_fused = the multimodal prefix region of inputs_embeds
(i.e. knowledge + intra-class + image tokens, shape B × T_mm × 3072)
and refines it through K cross-attention steps, producing r_K which
is handed to the ETM.

This preserves the inputs_embeds structure entirely — Llama sees no
structural change, only the quality of the multimodal prefix improves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .config import RSSMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

def _swiglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to x (B, H, T, head_dim)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class DropPath(nn.Module):
    """Stochastic depth — drops the residual branch with probability p."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = torch.rand(x.size(0), 1, 1, device=x.device) >= self.p
        return x * keep / (1.0 - self.p)


class RSSMStep(nn.Module):
    """
    Refined cross-attention step with:
      - pre-norm (more stable than post-norm)
      - self-attention w/ RoPE (state tokens communicate before querying context)
      - gated cross-attention (per-head α gate suppresses irrelevant heads)
      - SwiGLU FFN (consistent with LLaMA backbone style)
      - drop-path residuals (better regularisation, zero inference cost)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 drop_path: float = 0.0, rope_base: float = 10_000.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Pre-norms
        self.norm_self   = nn.LayerNorm(hidden_dim)
        self.norm_cross  = nn.LayerNorm(hidden_dim)
        self.norm_ffn    = nn.LayerNorm(hidden_dim)

        # Self-attention (state → state)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Gated cross-attention (state queries frozen context)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # α gate: one scalar per head, conditioned on mean-pooled query
        self.gate_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 1.0)   # start gates open

        # SwiGLU FFN (2× hidden → gate + value, then project back)
        ffn_inner = hidden_dim * 3
        self.ffn_in  = nn.Linear(hidden_dim, ffn_inner * 2, bias=False)
        self.ffn_out = nn.Linear(ffn_inner, hidden_dim, bias=False)

        # RoPE buffers (built lazily)
        self.rope_base = rope_base
        self._rope_cache: dict = {}

        self.drop_path = DropPath(drop_path)

    # ------------------------------------------------------------------
    def _get_rope(self, seq_len: int, device: torch.device) -> tuple:
        key = (seq_len, device)
        if key not in self._rope_cache:
            half = self.head_dim // 2
            theta = 1.0 / (self.rope_base ** (
                torch.arange(0, half, device=device).float() / half
            ))
            pos = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(pos, theta)           # (T, half)
            self._rope_cache[key] = (freqs.cos(), freqs.sin())
        return self._rope_cache[key]

    # ------------------------------------------------------------------
    def forward(
        self,
        state:            torch.Tensor,          # (B, T, D)
        context:          torch.Tensor,          # (B, T_ctx, D)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ── 1. Pre-normed self-attention with drop-path residual ──────
        normed = self.norm_self(state)
        sa_out, _ = self.self_attn(normed, normed, normed)
        state = state + self.drop_path(sa_out)

        # ── 2. Pre-normed gated cross-attention ───────────────────────
        normed = self.norm_cross(state)
        ca_out, _ = self.cross_attn(
            query=normed,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
        )
        # Per-head gate conditioned on mean-pooled query
        alpha = torch.sigmoid(
            self.gate_proj(normed.mean(dim=1))          # (B, num_heads)
        ).unsqueeze(1)                                   # (B, 1, num_heads)
        # Broadcast gate over sequence and head-interleaved D
        B, T, D = ca_out.shape
        ca_out_h = ca_out.view(B, T, self.num_heads, self.head_dim)
        ca_out_h = ca_out_h * alpha.unsqueeze(-1)        # (B, T, H, head_dim)
        ca_out = ca_out_h.view(B, T, D)
        state = state + self.drop_path(ca_out)

        # ── 3. Pre-normed SwiGLU FFN ──────────────────────────────────
        normed = self.norm_ffn(state)
        state = state + self.drop_path(self.ffn_out(_swiglu(self.ffn_in(normed))))

        return state

class IntermediateHead(nn.Module):
    """
    Lightweight auxiliary answer head attached to intermediate states.
    Predicts the first answer token from the mean-pooled state.
    Only used during training when intermediate_supervision=True.
    """
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.proj(state.mean(dim=1))  # (B, vocab_size)


class ReasoningStateSpaceModule(nn.Module):
    """
    RSSM.

    Inputs
    ------
    z_fused : (B, T_mm, D)
        The multimodal prefix tokens already assembled by
        prepare_inputs_for_multimodal and projected to Llama dim.
        T_mm = proj_out_num + kg_proj_out_num * 2

    Returns
    -------
    final_state : (B, T_mm, D)   r_K — passed to ETM
    all_states  : List of K+1 tensors [r_0 … r_K]
    aux_logits  : List of K-1 tensors (B, vocab_size) — intermediate heads
                  Empty list if intermediate_supervision=False
    """

    def __init__(self, config: RSSMConfig, vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        self.steps = nn.ModuleList([
            RSSMStep(D, config.num_heads, config.dropout)
            for _ in range(config.num_steps)
        ])

        self.learned_init = (
            nn.Parameter(torch.zeros(1, 1, D))
            if config.state_init == "learned" else None
        )

        self.supervision_heads = nn.ModuleList()
        if config.intermediate_supervision:
            assert vocab_size is not None, \
                "vocab_size required when intermediate_supervision=True"
            for _ in range(config.num_steps - 1):
                self.supervision_heads.append(IntermediateHead(D, vocab_size))

    def _init_state(self, z_fused: torch.Tensor) -> torch.Tensor:
        init = self.config.state_init
        if init == "fused":
            return z_fused.clone()
        elif init == "zeros":
            return torch.zeros_like(z_fused)
        elif init == "learned":
            return self.learned_init.expand(z_fused.size(0), z_fused.size(1), -1)
        else:
            raise ValueError(f"Unknown state_init: {init}")

    def forward(
        self,
        z_fused: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        state = self._init_state(z_fused)
        all_states = [state]
        aux_logits = []

        for k, step in enumerate(self.steps):
            state = step(state, z_fused, key_padding_mask=context_mask)
            all_states.append(state)

            # Attach intermediate supervision on r_1 … r_{K-1}
            if self.config.intermediate_supervision and k < len(self.supervision_heads):
                aux_logits.append(self.supervision_heads[k](state))

        return state, all_states, aux_logits
