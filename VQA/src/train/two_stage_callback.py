"""
train/two_stage_callback.py
===========================
Logging callback for RSSM + ETM integration.

All parameters that were trainable in the original PRIME-VQA
(LoRA adapters + projectors) remain trainable.  RSSM and ETM
are added on top as additional trainable modules.

No freezing is applied.  This callback only logs trainable
parameter counts at the start and end of training so you can
verify the parameter groups are correct.

HOW TO USE
----------
    from train.two_stage_callback import TwoStageCallback

    trainer = BaMCoVQATrainer(
        ...
        callbacks=[
            transformers.EarlyStoppingCallback(4),
            curriculum_callback,
            TwoStageCallback(model=model),
        ],
    )
"""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TwoStageCallback(TrainerCallback):
    """
    Logs trainable parameter breakdown at train start and end.
    No parameters are frozen or unfrozen — everything trainable
    in the original remains trainable; RSSM + ETM are added on top.

    Parameters
    ----------
    model   : the BaMCoLlamaForCausalLM (peft-wrapped) model
    stage1_epochs : accepted for API compatibility, has no effect
    verbose : print parameter counts
    """

    def __init__(self, model, stage1_epochs: int = None, verbose: bool = True):
        self.model = model
        self.verbose = verbose

    def _rprint(self, msg):
        if self.verbose:
            print(f"[TwoStageCallback] {msg}")

    def _log_params(self):
        groups = {"lora": 0, "projector": 0, "rssm": 0, "etm": 0, "other_trainable": 0}
        total = 0
        for name, param in self.model.named_parameters():
            n = param.numel()
            total += n
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                groups["lora"] += n
            elif "mm_projector" in name or "kg_projector" in name:
                groups["projector"] += n
            elif "rssm" in name:
                groups["rssm"] += n
            elif "etm" in name:
                groups["etm"] += n
            else:
                groups["other_trainable"] += n

        trainable = sum(groups.values())
        self._rprint(
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)"
        )
        for k, v in groups.items():
            if v > 0:
                self._rprint(f"  {k:<20} {v:>12,} params")

    def on_train_begin(self, args, state, control, **kwargs):
        self._rprint("Training begins — parameter groups:")
        self._log_params()

    def on_train_end(self, args, state, control, **kwargs):
        self._rprint("Training complete — final parameter groups:")
        self._log_params()