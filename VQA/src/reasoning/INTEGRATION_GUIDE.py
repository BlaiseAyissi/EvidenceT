"""
INTEGRATION GUIDE
=================
This file shows the exact minimal changes needed to the existing PRIME-VQA
files to activate RSSM + ETM.  No existing logic is removed.

Files to edit:
  1. VQA/src/model/BaMCo_VQA_arch.py   (3 changes)
  2. VQA/src/model/language_model/BaMCo_llama.py  (1 change)
  3. VQA/src/main.py  (2 changes)

================================================================================
CHANGE 1 — BaMCo_VQA_arch.py — import the mixin
================================================================================

# --- ADD at the top of BaMCo_VQA_arch.py, after existing imports ---
from reasoning.arch_mixin import PRIMEDiffuMixin
from reasoning.config import PRIMEDiffuConfig


================================================================================
CHANGE 2 — BaMCo_VQA_arch.py — inherit the mixin
================================================================================

# --- BEFORE ---
class BaMCoMetaForCausalLM(ABC):

# --- AFTER ---
class BaMCoMetaForCausalLM(PRIMEDiffuMixin, ABC):


================================================================================
CHANGE 3 — BaMCo_VQA_arch.py — replace prepare_inputs_for_multimodal
================================================================================

DELETE the entire existing prepare_inputs_for_multimodal method
(lines 2254-2334 in the combined output.txt).

The mixin (arch_mixin.py) provides the replacement implementation.
Python MRO ensures the mixin version is called automatically.


================================================================================
CHANGE 4 — BaMCo_llama.py — use composite loss in forward()
================================================================================

# --- BEFORE (in BaMCoLlamaForCausalLM.forward, after super().forward) ---
        return super().forward(
            input_ids=input_ids,
            ...
            labels=labels,
            ...
        )

# --- AFTER ---
        outputs = super().forward(
            input_ids=input_ids,
            ...
            labels=None,       # ← disable Llama's built-in CE
            ...
        )

        # Compute composite RSSM+ETM loss
        if self.has_reasoning_modules and labels is not None:
            loss, self._last_loss_breakdown = self.compute_reasoning_loss(
                lm_logits=outputs.logits,
                labels=labels,
            )
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs


================================================================================
CHANGE 5 — main.py — initialise reasoning modules after knowledge module
================================================================================

# --- ADD after this existing line ---
    if model_args.knowledge_encoder:
        model.get_model().initialize_knowledge_module(model_args=model_args)

# --- ADD IMMEDIATELY AFTER ---
    # Initialize RSSM + ETM reasoning modules
    reasoning_config = PRIMEDiffuConfig(
        rssm__num_steps=4,              # K: reasoning depth
        etm__num_evidence_tokens=32,    # L: soft prefix length
    )
    model.initialize_reasoning_modules(reasoning_config)
    rank0_print(f"Reasoning modules initialized: RSSM K={reasoning_config.rssm.num_steps}, "
                f"ETM L={reasoning_config.etm.num_evidence_tokens}")


================================================================================
CHANGE 6 — main.py — add reasoning params to LoRA ignore list (optional)
================================================================================

# In find_all_linear_names, add "rssm" and "etm" to ignore_keywords
# so LoRA doesn't wrap the new modules' linear layers:

    ignore_keywords = [
        'vision_tower', 'mm_projector', 'embed_tokens', 'lm_head',
        'seg_projector', 'seg_module',
        'rssm', 'etm',          # ← ADD THESE
    ]

# This ensures the RSSM and ETM are trained at their own LR (lr_rssm_etm)
# while LoRA adapters train the Llama backbone at the lower lr.


================================================================================
SUMMARY OF WHAT CHANGES vs. WHAT STAYS THE SAME
================================================================================

UNCHANGED:
  - BiomedCLIP image encoding (encode_images)
  - KGEncoder text/knowledge encoding (encode_knowledge)
  - GLIMS intra-class image encoding (retrieve_intra_class_images)
  - All three projectors (mm_projector, kg_projector, kg_projector_intra)
  - Llama forward pass and generation loop
  - BaMCoVQATrainer, DataCollator, Datasets, LoRA setup
  - All existing loss from labels (now goes through composite loss fn)
  - Curriculum learning callback and epoch pacing

ADDED (new code only):
  - reasoning/config.py      — all hyperparameters
  - reasoning/rssm.py        — K-step cross-attention refinement
  - reasoning/etm.py         — evidence distillation into L tokens
  - reasoning/loss.py        — composite 5-component loss
  - reasoning/arch_mixin.py  — integration glue

NEW EXECUTION FLOW:
  encode_images → encode_knowledge → retrieve_intra_class_images
      ↓
  RSSM: z_fused → [r_0, r_1, …, r_K]      (intermediate supervision)
      ↓
  ETM:  r_K + evidence_dict → evidence_tokens (L × 3072)
      ↓
  inputs_embeds: [BOS | evidence_L | kg | intra | img | question]
      ↓
  Llama autoregressive generation (UNCHANGED)
      ↓
  composite loss = LM + RSSM_inter + RSSM_cons + ETM_sal + ETM_div
"""
