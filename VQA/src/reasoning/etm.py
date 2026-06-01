"""
reasoning/etm.py
================
Evidence Transport Module (ETM)

In PRIME-VQA (BaMCo_VQA_arch.py), prepare_inputs_for_multimodal
produces three named feature tensors before building inputs_embeds:

    image_features      : (B, proj_out_num, 3072)
                          from encode_images → mm_projector
    knowledge_features  : (B, kg_proj_out_num, 3072)
                          from encode_knowledge → kg_projector
    intra_class_images  : (B, 1, kg_proj_out_num, 3072)  then unsqueeze(1)
                          from retrieve_intra_class_images → kg_projector_intra

The ETM:
  1. Assembles these into a context_bank
  2. Cross-attends L learned query tokens over context_bank,
     using r_K (final RSSM state) as a query bias
  3. Optionally gates each token by learned salience scores
  4. Returns L evidence tokens in Llama's embedding space (3072)
     to be prepended directly to inputs_embeds (replacing the raw
     multimodal tokens OR supplementing them as an additional prefix)

INSERTION STRATEGY
------------------
In prepare_inputs_for_multimodal, inputs_embeds is built as:
  [BOS | knowledge_features | intra_class_images | image_features | question_tokens]

We insert the ETM evidence tokens right after BOS, before the existing
multimodal tokens.  This gives Llama: 
  [BOS | ETM_evidence (L) | knowledge | intra-class | image | question]
This is additive — the original structure is preserved, the new tokens
provide a distilled "reasoning summary" that Llama can attend to.

No changes to Llama's generation loop are required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import ETMConfig


class SalienceGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.gate(tokens)          # (B, L, 1)
        return tokens * scores, scores


class EvidenceProjector(nn.Module):
    """Projects evidence tokens from hidden_dim → llama_hidden_dim."""
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        if num_layers == 1:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            mid = (in_dim + out_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(in_dim, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EvidenceTransportModule(nn.Module):
    """
    ETM.

    Inputs
    ------
    rssm_final_state : (B, T_mm, D)  r_K from RSSM
    evidence_dict    : {
        "image_features":     (B, proj_out_num, D),
        "knowledge_features": (B, kg_proj_out_num, D),
        "intra_class_images": (B, kg_proj_out_num, D),  ← after squeezing dim 1
    }

    Returns
    -------
    evidence_tokens : (B, L, D)   ready to insert into inputs_embeds
    gate_scores     : (B, L, 1) or None
    attn_weights    : (B, L, T_ctx)  for interpretability logging
    """

    def __init__(self, config: ETMConfig, fusion_dim: int):
        super().__init__()
        self.config = config
        D = fusion_dim

        # L learnable query slots
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_evidence_tokens, D) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(D)
        self.norm_out = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D * 4, D),
            nn.Dropout(config.dropout),
        )
        self.norm_ffn = nn.LayerNorm(D)

        self.salience_gate = SalienceGate(D) if config.salience_scoring else None

        # Since fusion_dim == llama_hidden_dim in PRIME-VQA (both 3072),
        # the projector is typically an identity or light MLP.
        self.projector = EvidenceProjector(
            in_dim=D,
            out_dim=config.llama_hidden_dim,
            num_layers=config.projection_layers,
            dropout=config.dropout,
        )

    def _build_context_bank(
        self, evidence_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Gather requested evidence streams, squeeze extra dims, concatenate.
        Handles the intra_class_images shape (B, 1, T, D) → (B, T, D).
        """
        parts = []
        for src in self.config.evidence_sources:
            if src not in evidence_dict:
                continue
            t = evidence_dict[src]
            # intra_class_images comes as (B, 1, T, D) from unsqueeze(1) in arch
            if t.dim() == 4:
                t = t.squeeze(1)   # → (B, T, D)
            parts.append(t)
        if not parts:
            raise ValueError(
                f"No evidence sources found. Requested: {self.config.evidence_sources}, "
                f"Available: {list(evidence_dict.keys())}"
            )
        return torch.cat(parts, dim=1)   # (B, T_ctx, D)

    def forward(
        self,
        rssm_final_state: torch.Tensor,
        evidence_dict: Dict[str, torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

        B = rssm_final_state.size(0)
        context_bank = self._build_context_bank(evidence_dict)   # (B, T_ctx, D)

        # Query tokens biased by mean-pooled RSSM final state (r_K)
        queries = self.query_tokens.expand(B, -1, -1).clone()
        rssm_bias = rssm_final_state.mean(dim=1, keepdim=True)   # (B, 1, D)
        queries = self.norm_q(queries + rssm_bias)

        attended, attn_weights = self.cross_attn(
            query=queries,
            key=context_bank,
            value=context_bank,
            key_padding_mask=context_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = self.norm_out(queries + attended)
        tokens = self.norm_ffn(tokens + self.ffn(tokens))

        gate_scores = None
        if self.salience_gate is not None:
            tokens, gate_scores = self.salience_gate(tokens)

        evidence_tokens = self.projector(tokens)   # (B, L, llama_hidden_dim)
        return evidence_tokens, gate_scores, attn_weights
