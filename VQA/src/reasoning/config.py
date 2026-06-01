"""
reasoning/config.py
===================
Configuration for the RSSM + ETM modules, grounded in the real
PRIME-VQA (BaMCo) architecture dimensions:

  - BiomedCLIP visual features  → 768-dim (ViT-B/16 output)
  - BiomedBERT knowledge text   → 512-dim (KGEncoder embed_dim)
  - kg_projector / kg_projector_intra output → configurable (proj_out_num tokens)
  - mm_projector output         → configurable (proj_out_num tokens)
  - Llama-3.2-3B hidden size    → 3072

All hyperparameters live here and nowhere else.
"""

from dataclasses import dataclass, field
from typing import Literal, List, Optional


@dataclass
class RSSMConfig:
    """
    Reasoning State Space Module.

    The RSSM takes z_fused (concatenation of image_features,
    knowledge_features, and intra_class_images already assembled
    inside prepare_inputs_for_multimodal) and progressively refines
    it through K cross-attention steps before handing off to the ETM.

    num_steps : int  (K)
        Number of refinement steps.  4 is the default — a sensible
        midpoint between compute cost and depth of reasoning.
        Range 2-8.

    hidden_dim : int
        Internal dimensionality of the RSSM. Must match fusion_dim
        (set automatically in PRIMEDiffuConfig.__post_init__).

    num_heads : int
        Attention heads per step.  Must divide hidden_dim.

    dropout : float
        Dropout inside each step's attention + FFN.

    state_init : "fused" | "zeros" | "learned"
        How r_0 is initialised.
        "fused"   = warm start from z_fused (recommended)
        "zeros"   = cold start
        "learned" = free nn.Parameter

    intermediate_supervision : bool
        Attach an auxiliary answer head to each r_1…r_{K-1}.
        Adds training signal at each step but requires answer_ids
        to be available in the batch (always true in PRIME-VQA).
    """
    num_steps: int = 4 #4
    hidden_dim: Optional[int] = None     # filled in by PRIMEDiffuConfig
    num_heads: int = 4
    dropout: float = 0.1
    state_init: Literal["fused", "zeros", "learned"] = "fused"
    intermediate_supervision: bool = True


@dataclass
class ETMConfig:
    """
    Evidence Transport Module.

    Takes the final RSSM state r_K and the raw feature streams
    (image patches, knowledge embeddings, intra-class image prototypes)
    and distils them into L soft-prompt tokens prepended to
    inputs_embeds before Llama's forward pass.

    num_evidence_tokens : int  (L)
        Number of soft-prompt tokens prepended to inputs_embeds.
        Range 8-64.  32 is a good default.

    evidence_sources : List[str]
        Which named feature tensors to attend over.
        Must correspond to keys in the evidence_dict built inside
        BaMCoMetaForCausalLM.prepare_inputs_for_multimodal.
        Valid: "image_features", "knowledge_features", "intra_class_images"

    salience_scoring : bool
        Learn a per-token gate weight in (0,1) to suppress irrelevant
        evidence tokens.  Adds a small auxiliary BCE loss.

    projection_layers : int (1 or 2)
        Depth of the projector from hidden_dim → llama_hidden_dim.
        2 (MLP) is recommended.

    llama_hidden_dim : int
        Must match Llama-3.2-3B hidden_size = 3072.
    """
    num_evidence_tokens: int = 16 # 32
    num_heads: int = 6 #8
    dropout: float = 0.1
    evidence_sources: List[str] = field(
        default_factory=lambda: ["image_features", "knowledge_features", "intra_class_images"]
    )
    salience_scoring: bool = True
    projection_layers: int = 2
    llama_hidden_dim: int = 3072


@dataclass
class RSSMLossConfig:
    """
    intermediate_weight : float
        Scale on the auxiliary CE loss from intermediate RSSM heads.

        IMPORTANT — keep this LOW relative to lm_loss_weight.
        If the aux losses are too strong they dominate the LM signal
        and cause the eval loss uptick seen in the first integration run.

        Recommended progression:
          Stage 1 (LoRA frozen):  0.05  — gentle; RSSM just warming up
          Stage 2 (joint):        0.1   — slightly stronger once stable
        Default here is the Stage 2 value; set lower if training Stage 1 only.

    consistency_weight : float
        Cosine-similarity smoothness penalty between consecutive states.
        0.01 is enough — this is a regulariser, not a primary signal.
    """
    intermediate_weight: float = 0.1
    consistency_weight: float = 0.01


@dataclass
class ETMLossConfig:
    """
    salience_loss_weight : float
        BCE loss on learned gate scores.
        Set to 0.0 unless you are computing explicit salience labels
        (e.g. from attention rollout).  Leaving it on without proper
        labels produces a noisy signal that hurts convergence.

    diversity_loss_weight : float
        Penalises pairwise cosine similarity between the L evidence tokens.
        0.02 is sufficient — increase only if you observe all L tokens
        collapsing to the same value in the prefix norm logs.
    """
    salience_loss_weight: float = 0.0   # disabled by default — needs explicit labels
    diversity_loss_weight: float = 0.02


@dataclass
class GenerationLossConfig:
    """
    lm_loss_weight : float
        Multiplier on the Llama CE loss.  Keep at 1.0.
        The LM loss must remain dominant — if auxiliary losses sum to
        more than ~20% of the LM loss the model optimises the wrong objective.

    answer_focus_weight : float
        Extra weight on answer tokens (tokens after question_len).
        PRIME-VQA already masks question tokens in labels (-100),
        so this only applies within the unmasked answer region.
        1.5 is safer than 2.0 during initial integration.
    """
    lm_loss_weight: float = 1.0
    answer_focus_weight: float = 1.5


@dataclass
class LossConfig:
    rssm: RSSMLossConfig = field(default_factory=RSSMLossConfig)
    etm: ETMLossConfig = field(default_factory=ETMLossConfig)
    generation: GenerationLossConfig = field(default_factory=GenerationLossConfig)


@dataclass
class TrainingStrategyConfig:
    """
    strategy : "freeze_llm" | "lora" | "full"
        Controls which Llama parameters are updated.
        PRIME-VQA already applies peft LoRA in main.py, so in practice
        you'd set strategy="freeze_llm" here (RSSM+ETM only) or
        "lora" to also adapt Llama's LoRA adapters.

    two_stage : bool
        Stage 1: freeze_llm, train only RSSM+ETM for two_stage_pretrain_steps.
        Stage 2: unfreeze LoRA, train jointly.
        Recommended: True — gives the new modules a warm start before
        Llama's gradients co-adapt.

    two_stage_pretrain_steps : int
        Steps for Stage 1.  ~10% of total steps is a good heuristic.
    """
    strategy: Literal["freeze_llm", "lora", "full"] = "freeze_llm" ## change this back to lora
    lora_rank: int = 64          # matches PRIME-VQA's existing lora_r=64
    lora_alpha: float = 16.0     # matches PRIME-VQA's lora_alpha=16
    warmup_steps: int = 500
    two_stage: bool = True
    two_stage_pretrain_steps: int = 2000


@dataclass
class EvalConfig:
    """
    metrics : List[str]
        Metrics to compute.  PRIME-VQA uses bleu, rouge1, accuracy_oe,
        accuracy_ce in its evaluater() function.  We extend with
        meteor, bertscore, f1.

    eval_reasoning_states : bool
        Log cosine-distance trajectory r_0→r_K per eval batch.
        Helps diagnose whether K is right-sized.

    eval_evidence_salience : bool
        Log mean gate scores from ETM.

    log_prefix_norms : bool
        Log L2 norms of evidence prefix tokens to detect collapse.
    """
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy_ce", "accuracy_oe", "bleu1", "rouge_l", "meteor", "bertscore", "f1"]
    )
    eval_reasoning_states: bool = True
    eval_evidence_salience: bool = True
    log_prefix_norms: bool = True


@dataclass
class PRIMEDiffuConfig:
    """
    Master config.

    fusion_dim : int
        Dimensionality of the fused multimodal representation coming
        out of prepare_inputs_for_multimodal.

        In PRIME-VQA this is 3072 (Llama hidden size), because
        image_features, knowledge_features, and intra_class_images
        are all projected to 3072 via mm_projector / kg_projector
        before being concatenated into inputs_embeds.

        The RSSM and ETM therefore operate natively in this space,
        and no extra projection is needed on the input side.

    seed : int
        Global random seed.
    """
    fusion_dim: int = 3072       # Llama-3.2-3B hidden_size = 3072
    seed: int = 25               # matches PRIME-VQA default seed

    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    etm: ETMConfig = field(default_factory=ETMConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingStrategyConfig = field(default_factory=TrainingStrategyConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        # RSSM hidden_dim auto-inherits fusion_dim if not explicitly set
        if self.rssm.hidden_dim is None:
            self.rssm.hidden_dim = self.fusion_dim
        # ETM llama_hidden_dim should match fusion_dim since features
        # are already in Llama's embedding space
        self.etm.llama_hidden_dim = self.fusion_dim