"""Shared Trainer plumbing for the MLM and NLI training stages.

Extracted from scripts/mlm/train_roberta.py so every training stage gets identical checkpoint hygiene:
a *validated* resume, a rotation-proof copy of the metric history, and durable end-of-run artifacts.
All three exist because the first MLM run lost 420,000 steps of training to a checkpoint that looked
fine from its directory name and wasn't.
"""

import json
import math
import os
import re

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

_CHECKPOINT_DIR_RE = re.compile(r'checkpoint-(\d+)$')
_WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')


def find_resumable_checkpoint(output_dir):
    """Highest-numbered checkpoint that can actually be resumed from.

    get_last_checkpoint() only pattern-matches the `checkpoint-<step>` directory NAME, so it happily
    returns a checkpoint whose contents are incomplete and the resume then fails deep inside Trainer.
    This bit us for real: of the first MLM run's six checkpoints, only 128000 still had its weights and
    optimizer state — 130000 had weights only, and 132000/518000/546000/548000 had neither, yet
    get_last_checkpoint() would have picked 548000.

    Both files are required, not just the weights: resuming from a weights-only checkpoint makes
    Trainer silently reinitialize the optimizer moments and restart the LR schedule from step 0,
    which is a quiet quality regression rather than a visible error.
    """
    if not os.path.isdir(output_dir):
        return None

    resumable = []
    for entry in sorted(os.scandir(output_dir), key=lambda e: e.name):
        match = _CHECKPOINT_DIR_RE.match(entry.name)
        if not (entry.is_dir() and match):
            continue
        has_weights = any(os.path.isfile(os.path.join(entry.path, w)) for w in _WEIGHT_FILES)
        has_optimizer = os.path.isfile(os.path.join(entry.path, 'optimizer.pt'))

        missing = []
        if not has_weights:
            missing.append(' or '.join(_WEIGHT_FILES))
        if not has_optimizer:
            missing.append('optimizer.pt')

        if missing:
            print(f"Skipping unusable checkpoint {entry.name}: missing {', '.join(missing)}")
        else:
            resumable.append((int(match.group(1)), entry.path))

    return max(resumable)[1] if resumable else None


def resolve_resume_checkpoint(output_dir):
    """What to hand Trainer.train(resume_from_checkpoint=...), or None to train from scratch.

    Wraps find_resumable_checkpoint() with the two things every caller needs: a loud warning when the
    newest checkpoint on disk had to be skipped (silence there would hide real data loss), and the
    torch>=2.6 gate that Transformers 5.15+ enforces before it will load optimizer state.
    """
    newest_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    last_checkpoint = find_resumable_checkpoint(output_dir)
    if newest_checkpoint and last_checkpoint and os.path.normpath(newest_checkpoint) != os.path.normpath(last_checkpoint):
        print(
            f"NOTE: newest checkpoint on disk is {os.path.basename(newest_checkpoint)} but it is not resumable; "
            f"falling back to {os.path.basename(last_checkpoint)}"
        )

    torch_version_parts = torch.__version__.split('+')[0].split('.')
    torch_major = int(torch_version_parts[0])
    torch_minor = int(torch_version_parts[1]) if len(torch_version_parts) > 1 else 0
    can_resume_checkpoint = (torch_major, torch_minor) >= (2, 6)

    if last_checkpoint and can_resume_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
    elif last_checkpoint and not can_resume_checkpoint:
        print(
            f"Checkpoint {os.path.basename(last_checkpoint)} found but skipping resume because this "
            f"environment uses torch {torch.__version__} (<2.6); starting from scratch to avoid blocked "
            "optimizer-state loading. Upgrade torch to resume instead of restarting."
        )
        last_checkpoint = None
    else:
        print("No checkpoint found, starting training from scratch")

    return last_checkpoint


class RunRootStateWriter(TrainerCallback):
    """Mirrors trainer_state.json to the run root on every save.

    Trainer writes trainer_state.json only *inside* each checkpoint, so save_total_limit rotation deletes the
    metric history along with the weights, and an interrupted run leaves no history at the run root at all.
    That is precisely how the first MLM run's segment-1 numbers came within one deleted directory of being
    lost: they survived only because checkpoint-128000 happened to be the one that kept its copy.

    Cost is negligible — the file is overwritten in place, so it occupies ~0.7 MB at the end of a 2.88M-step
    schedule no matter how many times it is written (vs ~800 MB per checkpoint).
    """

    def __init__(self, run_dir):
        self.state_path = run_dir / 'trainer_state.json'

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            state.save_to_json(str(self.state_path))


def save_run_artifacts(trainer, run_dir, metrics, metrics_filename='final_metrics.json'):
    """Persist final metrics and the full log history to the RUN ROOT, not inside a checkpoint.

    Metrics used to be print-only, so the only durable record of any number was the trainer_state.json
    that Trainer drops inside each checkpoint — which save_total_limit rotates away. Both files written
    here are small JSON and sit outside the .gitignore'd checkpoint-*/final globs, so they get committed
    as the permanent record of the run.
    """
    with open(run_dir / metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    trainer.state.save_to_json(str(run_dir / 'trainer_state.json'))  # every logged loss, LR and eval metric
    print(f"Metrics and full log history saved to {run_dir}")


def export_final_model(trainer, tokenizer, run_dir, subdir='final'):
    """Standalone weights-only export: the artifact downstream stages consume.

    Step checkpoints are subject to save_total_limit rotation and carry optimizer state (~3x the size),
    so they are the wrong thing to hand to the next stage or to upload anywhere.
    """
    final_dir = run_dir / subdir
    trainer.save_model(str(final_dir))          # config.json + model.safetensors
    tokenizer.save_pretrained(str(final_dir))   # tokenizer.json + tokenizer_config.json + special_tokens_map.json
    print(f"Final model + tokenizer saved to {final_dir}")
    return final_dir


def add_perplexity(metrics, loss_key='eval_loss'):
    """Attach perplexity next to a cross-entropy loss, for metrics where it is the number worth quoting."""
    if loss_key in metrics:
        metrics['perplexity'] = math.exp(metrics[loss_key])
    return metrics
