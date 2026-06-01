"""
reasoning/arch_mixin.py
=======================
Drop-in mixin for BaMCoMetaForCausalLM (BaMCo_VQA_arch.py).

HOW TO INTEGRATE
----------------
In VQA/src/model/BaMCo_VQA_arch.py, add two imports at the top:

    from reasoning.arch_mixin import PRIMEDiffuMixin
    from reasoning.config import PRIMEDiffuConfig

Then change:

    class BaMCoMetaForCausalLM(ABC):
        ...

to:

    class BaMCoMetaForCausalLM(PRIMEDiffuMixin, ABC):
        ...

That's the only edit needed to BaMCo_VQA_arch.py.

HOW IT WORKS
------------
PRIMEDiffuMixin:
  1. Adds initialize_reasoning_modules(config) — call once after
     initialize_knowledge_module() in main.py.
  2. Overrides prepare_inputs_for_multimodal to inject RSSM + ETM
     between the existing feature encoding and the inputs_embeds
     assembly.  The rest of the method is identical to the original.
  3. Overrides forward (via _reasoning_forward_hook) to collect the
     RSSM/ETM auxiliary outputs and add them to the Llama CE loss.

The overridden prepare_inputs_for_multimodal captures the three
named feature tensors (image_features, knowledge_features,
intra_class_images) right after they are computed, runs RSSM + ETM,
then inserts the L evidence tokens right after BOS in inputs_embeds:

  BEFORE:  [BOS | kg | intra | img | question_tokens]
  AFTER:   [BOS | ETM_evidence(L) | kg | intra | img | question_tokens]

This is purely additive — original multimodal tokens are preserved.
Llama's autoregressive generation is not modified in any way.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

from .config import PRIMEDiffuConfig
from .rssm import ReasoningStateSpaceModule
from .etm import EvidenceTransportModule
from .loss import PRIMEDiffuLoss


class PRIMEDiffuMixin:
    """
    Mixin to be added to BaMCoMetaForCausalLM.
    Provides reasoning module init, the patched prepare_inputs_for_multimodal,
    and the composite loss computation.
    """

    # ------------------------------------------------------------------ #
    # Initialisation — call once in main.py after initialize_knowledge_module
    # ------------------------------------------------------------------ #

    def initialize_reasoning_modules(self, config: PRIMEDiffuConfig):
        """
        Instantiate RSSM, ETM, and composite loss.
        Call this in main.py right after:
            model.get_model().initialize_knowledge_module(model_args)

        Example:
            reasoning_config = PRIMEDiffuConfig()
            model.initialize_reasoning_modules(reasoning_config)
        """
        self._reasoning_config = config

        vocab_size = getattr(self, 'vocab_size', None) or \
                     getattr(self.get_model().config, 'vocab_size', 32000)

        rssm = ReasoningStateSpaceModule(
            config=config.rssm,
            vocab_size=vocab_size if config.rssm.intermediate_supervision else None,
        ).to(self.device)

        etm = EvidenceTransportModule(
            config=config.etm,
            fusion_dim=config.fusion_dim,
        ).to(self.device)

        # Register on both self (outer/peft model) and get_model() (inner model)
        # so has_reasoning_modules and _get_rssm/_get_etm work regardless of
        # which object prepare_inputs_for_multimodal is dispatched to via MRO.
        self.rssm = rssm
        self.etm  = etm
        try:
            inner = self.get_model()
            inner.rssm = rssm   # same object — no extra memory
            inner.etm  = etm
        except Exception:
            print("------------passing without rssm and etm ---------------------")
            pass

        self._reasoning_loss_fn = PRIMEDiffuLoss(config.loss)

        # Storage for passing RSSM/ETM outputs from prepare_inputs to forward
        self._rssm_outputs = None
        self._etm_outputs = None

        print(f"[PRIMEDiffuMixin] RSSM (K={config.rssm.num_steps}) and "
              f"ETM (L={config.etm.num_evidence_tokens}) initialized.")

    @property
    def has_reasoning_modules(self) -> bool:
        # RSSM and ETM may be on self (outer/peft model) OR on self.get_model()
        # (inner BaMCoMetaModel), depending on where initialize_reasoning_modules
        # was called from. Check both so the property works regardless.
        if hasattr(self, 'rssm') and hasattr(self, 'etm'):
            return True
        try:
            inner = self.get_model()
            return hasattr(inner, 'rssm') and hasattr(inner, 'etm')
        except Exception:
            return False

    def _get_rssm(self):
        if hasattr(self, 'rssm'):
            return self.rssm
        return self.get_model().rssm

    def _get_etm(self):
        if hasattr(self, 'etm'):
            return self.etm
        return self.get_model().etm

    # ------------------------------------------------------------------ #
    # Patched prepare_inputs_for_multimodal
    # ------------------------------------------------------------------ #

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        term_list,
    ):
        """
        Replacement for BaMCoMetaForCausalLM.prepare_inputs_for_multimodal.

        Steps 1-4 are identical to the original.
        Step 5 (RSSM + ETM) is inserted between feature encoding
        and inputs_embeds assembly.
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # ---- Step 1: encode images (identical to original) ----
        image_features, RAG_query_image_features = self.encode_images(images)

        # ---- Step 2: encode knowledge + intra-class (identical) ----
        knowledge_features = None
        intra_class_images = None
        # Pre-compute validity here so Step 2 and Step 4 use the same logic.
        def _is_valid_term_list(tl):
            if tl is None:
                return False
            if isinstance(tl, str):
                return tl != "None"
            if hasattr(tl, '__iter__'):
                items = list(tl)
                return len(items) > 0 and all(t is not None and t != "None" for t in items)
            return False

        term_list_valid = _is_valid_term_list(term_list)

        if term_list_valid:
            intra_class_images = self.retrieve_intra_class_images(
                RAG_query_image_features
            ).unsqueeze(1)
            knowledge_features = self.encode_knowledge(term_list).unsqueeze(1)

        # ---- Step 3: get base token embeddings (identical) ----
        if "llama" in self.get_model().name_or_path or "Llama" in self.get_model().name_or_path:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        else:
            inputs_embeds = self.get_model().base_model.wte(input_ids)

        # ---- Step 4 (NEW): RSSM + ETM ----
        """print('[PRIMEDiffuMixin] term_list_valid:', term_list_valid,
              '| type:', type(term_list).__name__,
              '| value:', repr(str(term_list)[:80]))"""

        if self.has_reasoning_modules and term_list_valid:
            # Build z_fused: concatenate all multimodal features
            # knowledge_features: (B, 1, T_kg, D) → squeeze → (B, T_kg, D)
            # intra_class_images: (B, 1, T_intra, D) → squeeze → (B, T_intra, D)
            kg_flat     = knowledge_features      # (B, T_kg, D)
            intra_flat  = intra_class_images     # (B, T_intra, D)
            # image_features is already (B, proj_out_num, D)
            z_fused = torch.cat([kg_flat, intra_flat, image_features], dim=1)
            # z_fused: (B, T_mm, D)

            # Run RSSM — refine the multimodal representation
            rssm_final, all_states, aux_logits = self._get_rssm()(z_fused)

            # Run ETM — distil into L evidence tokens
            evidence_dict = {
                "image_features":     image_features,           # (B, T_img, D)
                "knowledge_features": kg_flat,                  # (B, T_kg, D)
                "intra_class_images": intra_flat,               # (B, T_intra, D)
            }
            evidence_tokens, gate_scores, attn_weights = self._get_etm()(
                rssm_final_state=rssm_final,
                evidence_dict=evidence_dict,
            )
            # evidence_tokens: (B, L, 3072)

            # Stash for loss computation in forward()
            self._rssm_outputs = (all_states, aux_logits)
            self._etm_outputs  = (evidence_tokens, gate_scores)

            # ---- Step 5: assemble inputs_embeds WITH evidence prefix ----
            # [BOS | evidence(L) | kg | intra | img | question_tokens]
            bos_embed   = inputs_embeds[:, :1, :]
            rest_embeds = inputs_embeds[:, (
                knowledge_features.shape[1] +
                intra_class_images.shape[1] +
                image_features.shape[1] + 1
            ):, :]

            inputs_embeds = torch.cat(
                (
                    bos_embed,
                    evidence_tokens,          # ← NEW: L evidence tokens
                    knowledge_features,        # original kg tokens
                    intra_class_images,        # original intra tokens
                    image_features,            # original image tokens
                    rest_embeds,               # question + padding tokens
                ),
                dim=1,
            )

            # Extend attention_mask for the L new evidence tokens
            if attention_mask is not None:
                L = evidence_tokens.size(1)
                evidence_attn = torch.ones(
                    attention_mask.size(0), L,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    [attention_mask[:, :1], evidence_attn, attention_mask[:, 1:]],
                    dim=1,
                )

            # Extend labels to ignore evidence prefix tokens (-100)
            if labels is not None:
                L = evidence_tokens.size(1)
                evidence_labels = torch.full(
                    (labels.size(0), L), -100,
                    dtype=labels.dtype, device=labels.device,
                )
                labels = torch.cat(
                    [labels[:, :1], evidence_labels, labels[:, 1:]],
                    dim=1,
                )

        else:
            # ---- Fallback: original assembly (no knowledge or no reasoning) ----
            self._rssm_outputs = None
            self._etm_outputs  = None

            if term_list_valid:
                inputs_embeds = torch.cat(
                    (
                        inputs_embeds[:, :1, :],
                        knowledge_features,
                        intra_class_images,
                        image_features,
                        inputs_embeds[:, (
                            knowledge_features.shape[1] +
                            intra_class_images.shape[1] +
                            image_features.shape[1] + 1
                        ):, :],
                    ),
                    dim=1,
                )
            else:
                inputs_embeds = torch.cat(
                    (
                        inputs_embeds[:, :1, :],
                        image_features,
                        inputs_embeds[:, (image_features.shape[1] + 1):, :],
                    ),
                    dim=1,
                )

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    # ------------------------------------------------------------------ #
    # Composite loss helper — call from forward() in BaMCoLlamaForCausalLM
    # ------------------------------------------------------------------ #

    def compute_reasoning_loss(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        answer_mask: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        salience_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the full composite loss.

        Call this from BaMCoLlamaForCausalLM.forward() INSTEAD of the
        standard Llama loss, then return it as the loss field.

        Example replacement in BaMCoLlamaForCausalLM.forward():

            # --- REPLACE ---
            outputs = super().forward(...)
            loss = outputs.loss

            # --- WITH ---
            outputs = super().forward(..., labels=None)   # disable Llama CE
            if self.has_reasoning_modules:
                loss, breakdown = self.compute_reasoning_loss(
                    lm_logits=outputs.logits,
                    labels=labels_shifted,      # pass the combined labels here
                    answer_ids=batch.get("answer_ids"),
                )
                outputs = outputs.__class__(loss=loss, **{k:v for k,v in outputs.__dict__.items() if k != 'loss'})
        """
        # No labels during inference — return zero loss, don't enter loss.py
        if labels is None:
            zero = torch.tensor(0.0, device=lm_logits.device)
            return zero, {"lm_loss": 0.0, "total_loss": 0.0}

        rssm_all_states, rssm_aux_logits = (
            self._rssm_outputs if self._rssm_outputs is not None else (None, None)
        )
        evidence_tokens, gate_scores = (
            self._etm_outputs if self._etm_outputs is not None else (None, None)
        )

        total, breakdown = self._reasoning_loss_fn(
            lm_logits=lm_logits,
            labels=labels,
            answer_mask=answer_mask,
            rssm_aux_logits=rssm_aux_logits or [],
            rssm_all_states=rssm_all_states or [],
            answer_ids=answer_ids,
            evidence_tokens=evidence_tokens,
            gate_scores=gate_scores,
            salience_labels=salience_labels,
        )
        return total, breakdown

    # ------------------------------------------------------------------ #
    # Parameter groups for optimiser (call from main.py if desired)
    # ------------------------------------------------------------------ #

    def get_reasoning_parameter_groups(
        self,
        lr_rssm_etm: float = 1e-4,
    ) -> List[Dict]:
        """
        Returns a list of parameter groups for AdamW that separates
        RSSM+ETM params from everything else.  Pass this to the
        optimizer instead of model.parameters().

        Usage in main.py (optional — you can also let Trainer handle it):
            if model.has_reasoning_modules:
                param_groups = model.get_reasoning_parameter_groups()
                optimizer = torch.optim.AdamW(param_groups)
        """
        if not self.has_reasoning_modules:
            return [{"params": self.parameters()}]

        rssm_etm_ids = set(
            id(p) for p in list(self.rssm.parameters()) + list(self.etm.parameters())
        )
        other_trainable = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in rssm_etm_ids
        ]
        return [
            {
                "params": list(self.rssm.parameters()) + list(self.etm.parameters()),
                "lr": lr_rssm_etm,
                "name": "rssm_etm",
            },
            {
                "params": other_trainable,
                "name": "base_model",
            },
        ]