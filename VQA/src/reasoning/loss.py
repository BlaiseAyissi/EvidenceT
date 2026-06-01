"""
reasoning/loss.py
=================
Composite loss for PRIME-VQA + RSSM/ETM integration.

Combines:
  1. Primary LM loss  — Llama next-token CE (with optional answer focus)
  2. RSSM intermediate supervision — auxiliary CE per intermediate state
  3. RSSM consistency — smoothness prior across reasoning trajectory
  4. ETM salience — BCE on gate scores (optional)
  5. ETM diversity — penalise evidence token collapse (optional)

All weights come from LossConfig — set any weight to 0.0 to disable
that component completely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import LossConfig


class PRIMEDiffuLoss(nn.Module):

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------ #
    # 1. Primary generation loss
    # ------------------------------------------------------------------ #

    def _lm_loss(
        self,
        logits: torch.Tensor,                          # (B, T, V)
        labels: torch.Tensor,                          # (B, T)
        answer_mask: Optional[torch.Tensor] = None,   # (B, T) bool — True on answer tokens
    ) -> torch.Tensor:
        cfg = self.config.generation
        B, T, V = logits.shape

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        if answer_mask is not None:
            shift_mask = answer_mask[:, 1:].float()
            weights = 1.0 + shift_mask * (cfg.answer_focus_weight - 1.0)
            per_tok = F.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1),
                reduction="none", ignore_index=-100
            )
            valid = (shift_labels.view(-1) != -100).float()
            loss = (per_tok * weights.view(-1) * valid).sum() / valid.sum().clamp(min=1)
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
            )
        return cfg.lm_loss_weight * loss

    # ------------------------------------------------------------------ #
    # 2. RSSM intermediate supervision
    # ------------------------------------------------------------------ #

    def _rssm_intermediate_loss(
        self,
        aux_logits: List[torch.Tensor],   # each (B, vocab_size)
        answer_ids: torch.Tensor,          # (B,)  first answer token
    ) -> torch.Tensor:
        if not aux_logits or self.config.rssm.intermediate_weight == 0.0:
            return torch.tensor(0.0, device=answer_ids.device if answer_ids is not None else torch.device("cpu"))
        losses = [F.cross_entropy(logit, answer_ids) for logit in aux_logits]
        return self.config.rssm.intermediate_weight * torch.stack(losses).mean()

    # ------------------------------------------------------------------ #
    # 3. RSSM consistency
    # ------------------------------------------------------------------ #

    def _rssm_consistency_loss(self, all_states: List[torch.Tensor]) -> torch.Tensor:
        w = self.config.rssm.consistency_weight
        if w == 0.0 or len(all_states) < 2:
            return torch.tensor(0.0)
        penalties = []
        for s_prev, s_next in zip(all_states[:-1], all_states[1:]):
            p = s_prev.mean(dim=1)
            n = s_next.mean(dim=1)
            penalties.append((1.0 - F.cosine_similarity(p, n, dim=-1)).mean())
        return w * torch.stack(penalties).mean()

    # ------------------------------------------------------------------ #
    # 4. ETM salience
    # ------------------------------------------------------------------ #

    def _etm_salience_loss(
        self,
        gate_scores: Optional[torch.Tensor],      # (B, L, 1)
        salience_labels: Optional[torch.Tensor],  # (B, L) binary
    ) -> torch.Tensor:
        w = self.config.etm.salience_loss_weight
        if w == 0.0 or gate_scores is None or salience_labels is None:
            return torch.tensor(0.0)
        return w * F.binary_cross_entropy(
            gate_scores.squeeze(-1), salience_labels.float()
        )

    # ------------------------------------------------------------------ #
    # 5. ETM diversity
    # ------------------------------------------------------------------ #

    def _etm_diversity_loss(self, evidence_tokens: torch.Tensor) -> torch.Tensor:
        w = self.config.etm.diversity_loss_weight
        if w == 0.0:
            return torch.tensor(0.0)
        normed = F.normalize(evidence_tokens, dim=-1)
        gram = torch.bmm(normed, normed.transpose(1, 2))
        L = gram.size(1)
        eye = torch.eye(L, device=gram.device, dtype=torch.bool).unsqueeze(0)
        off_diag = gram.masked_fill(eye, 0.0)
        return w * (off_diag ** 2).sum(dim=(1, 2)).mean() / max(L * (L - 1), 1)

    # ------------------------------------------------------------------ #
    # Master forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: Optional[torch.Tensor],           # None during inference
        answer_mask: Optional[torch.Tensor] = None,
        rssm_aux_logits: Optional[List[torch.Tensor]] = None,
        rssm_all_states: Optional[List[torch.Tensor]] = None,
        answer_ids: Optional[torch.Tensor] = None,
        evidence_tokens: Optional[torch.Tensor] = None,
        gate_scores: Optional[torch.Tensor] = None,
        salience_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = lm_logits.device

        # During inference labels is None — return zero loss immediately.
        # compute_reasoning_loss should not be called in that path, but guard
        # here as a safety net so loss.py never crashes on None labels.
        if labels is None:
            zero = torch.tensor(0.0, device=device)
            return zero, {"lm_loss": 0.0, "rssm_intermediate_loss": 0.0,
                          "rssm_consistency_loss": 0.0, "etm_salience_loss": 0.0,
                          "etm_diversity_loss": 0.0, "total_loss": 0.0}

        lm      = self._lm_loss(lm_logits, labels, answer_mask)
        rssm_i  = self._rssm_intermediate_loss(
            rssm_aux_logits or [],
            answer_ids if answer_ids is not None
            else torch.zeros(lm_logits.size(0), dtype=torch.long, device=device)
        ).to(device)
        rssm_c  = self._rssm_consistency_loss(rssm_all_states or []).to(device)
        etm_sal = self._etm_salience_loss(gate_scores, salience_labels).to(device)
        etm_div = (
            self._etm_diversity_loss(evidence_tokens) if evidence_tokens is not None
            else torch.tensor(0.0, device=device)
        )

        total = lm + rssm_i + rssm_c + etm_sal + etm_div

        breakdown = {
            "lm_loss":                  lm.item(),
            "rssm_intermediate_loss":   rssm_i.item(),
            "rssm_consistency_loss":    rssm_c.item(),
            "etm_salience_loss":        etm_sal.item(),
            "etm_diversity_loss":       etm_div.item(),
            "total_loss":               total.item(),
        }
        return total, breakdown