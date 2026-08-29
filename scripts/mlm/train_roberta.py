from tokenizers import Tokenizer
from transformers import RobertaConfig, EarlyStoppingCallback, RobertaForMaskedLM, DataCollatorForLanguageModeling, PreTrainedTokenizerFast, Trainer, TrainerCallback, TrainingArguments
import json
import math
import os
import re
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from paths import Paths
from tokenizer.core.constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import torch

paths = Paths()

tokenizer_path = paths.tokenizer_name_generator('03_sandhi_codepoint_bert') / 'tokenizer.json'  # confirmed lowest fertility (1.341) of all 12 variants

# Single load, reused for config sizing below and for encode_batch further down (was loaded twice before)
raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))

vocab_size = raw_tokenizer.get_vocab_size()          # already an int; no len() needed
pad_id = raw_tokenizer.token_to_id(PAD_TOKEN)         # looked up once, reused for pad_token_id and max_position_embeddings below

config = RobertaConfig(
    vocab_size=vocab_size,                             # matches this tokenizer's trained vocab (32000)
    max_position_embeddings=512 + pad_id + 1,          # seq_len + pad_id + 1: covers RoBERTa's padding-offset position ids (computed dynamically, not hardcoded, since pad_id can differ per tokenizer)
    num_attention_heads=12,                            # 768/12 = 64 dim/head, standard ratio
    num_hidden_layers=6,                               # DistilRoBERTa-sized depth, fits your GPU budget
    type_vocab_size=1,                                 # RoBERTa-style: no NSP/segment ids, single input type
    pad_token_id=pad_id,                               # reuse looked-up id
    bos_token_id=raw_tokenizer.token_to_id(BOS_TOKEN),  # [CLS]-equivalent
    eos_token_id=raw_tokenizer.token_to_id(EOS_TOKEN),  # [SEP]-equivalent
    add_cross_attention=False                          # encoder-only; redundant with is_decoder default False but kept explicit
)

model = RobertaForMaskedLM(config=config)
print(f"Model initialized with vocab size {vocab_size} and max position embeddings {config.max_position_embeddings}")

# MLM configuration — wrap the same raw_tokenizer once for HF-side tooling (Trainer, DataCollator)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                    cls_token=BOS_TOKEN,
                                    sep_token=EOS_TOKEN,
                                    pad_token=PAD_TOKEN,
                                    unk_token=UNK_TOKEN,
                                    mask_token=MASK_TOKEN)

print(f"Tokenizer vocab size: {config.vocab_size}, pad token id: {tokenizer.pad_token_id}, bos token id: {tokenizer.cls_token_id}, eos token id: {tokenizer.sep_token_id}, unk token id: {tokenizer.unk_token_id}, mask token id: {tokenizer.mask_token_id}")
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

# Load dataset
data_files = {
    "train": str(paths.train_sandhi_marked),
    'test': str(paths.test_sandhi_marked)
}

data = load_dataset('text', data_files=data_files)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training configuration
mlm_run_dir = paths.mlm_run_generator('03_sandhi_codepoint_bert')  # resolved once; reused for output_dir and for the final export below

training_args = TrainingArguments(
    output_dir=str(mlm_run_dir),            # named per tokenizer + architecture, avoids checkpoint collisions across runs
    eval_strategy='steps',                  # CHANGED from 'epoch': must match save_strategy for load_best_model_at_end (needed by EarlyStoppingCallback) — see eval_subset below for why this is cheap
    eval_steps=10000,                       # kept equal to save_steps so every save has a fresh eval_loss to compare against
    save_strategy='steps',                  # CHANGED from 'epoch': one epoch is ~479k steps here, far too coarse for crash recovery
    save_steps=10000,                       # RAISED from 2000: each checkpoint is ~800 MB (weights + optimizer state), so 2000 meant ~240 saves and ~190 GB per epoch — unmanageable to retain or transfer. ~48 saves/epoch still bounds crash loss to ~2% of an epoch
    num_train_epochs=6,                     # ceiling, not a target; early stopping + load_best_model_at_end guard against overshooting
    learning_rate=1e-4,                     # standard from-scratch RoBERTa/BERT peak LR
    lr_scheduler_type='linear',             # linear decay after warmup; matches original RoBERTa recipe
    warmup_ratio=0.06,                      # ~6% of steps ramp-up; restored, was missing here — important for a randomly-initialized transformer
    weight_decay=0,                         # no weight decay; consider ~0.01 later if eval loss plateaus/overfits
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,          # effective batch size = 16*4 = 64
    fp16=True,                              # mixed precision for GPU throughput/memory headroom
    save_total_limit=3,                     # bumped from 2: a little more resume margin with frequent step-based saves
    load_best_model_at_end=True,            # restored: required by EarlyStoppingCallback to track state.best_metric
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

# Callbacks
# patience is in EVAL CALLS, not epochs — at eval_steps=10000 that's a ~150,000-step (~0.3 epoch) no-improvement
# window. Left at 15 deliberately: too loose only wastes compute, too tight can kill a run that was still improving.
# Drop to ~5 (a ~50k-step window) once you've seen how noisy eval_loss actually is step-to-step.
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=15)


class RunRootStateWriter(TrainerCallback):
    """Mirrors trainer_state.json to the run root on every save.

    Trainer writes trainer_state.json only *inside* each checkpoint, so save_total_limit rotation deletes the
    metric history along with the weights, and an interrupted run leaves no history at the run root at all.
    That is precisely how this run's segment-1 numbers came within one deleted directory of being lost: they
    survived only because checkpoint-128000 happened to be the one that kept its copy.

    Cost is negligible — the file is overwritten in place, so it occupies ~0.7 MB at the end of the full
    2.88M-step schedule no matter how many times it is written (vs ~800 MB per checkpoint).
    """

    def __init__(self, run_dir):
        self.state_path = run_dir / 'trainer_state.json'

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            state.save_to_json(str(self.state_path))


run_root_state_writer = RunRootStateWriter(mlm_run_dir)

# DataLoading
raw_tokenizer.enable_truncation(max_length=512)  # required: your token-length histogram has outlier lines up to ~2500 tokens, which would exceed max_position_embeddings without this

def tokenize_function(examples):
    encoding = raw_tokenizer.encode_batch(examples['text'])
    return {'input_ids': [e.ids for e in encoding],
            'attention_mask': [e.attention_mask for e in encoding]}

tokenized_datasets = data.map(tokenize_function, batched=True, remove_columns=['text'])

# Frequent step-based eval (every 2000 steps) against the FULL 3.4M-line test set would mean ~213k
# eval forward-passes per pass (3.4M / eval_batch_size 16) — almost as expensive as a training epoch itself,
# repeated ~240 times per epoch. Use a fixed, shuffled subsample for these frequent checks instead; the full
# test set is still evaluated once at the very end, below, for the real reported metric.
eval_subset_size = min(20000, len(tokenized_datasets['test']))
eval_subset = tokenized_datasets['test'].shuffle(seed=42).select(range(eval_subset_size))

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=eval_subset,          # small shuffled subsample — see comment above; full test set evaluated separately at the end
    data_collator=data_collator,
    callbacks=[early_stopping_callback,  # was defined but never attached before — early stopping wasn't actually active
               run_root_state_writer]    # keeps a rotation-proof, interruption-proof copy of the metric history
)

_CHECKPOINT_DIR_RE = re.compile(r'checkpoint-(\d+)$')
_WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')


def find_resumable_checkpoint(output_dir):
    """Highest-numbered checkpoint that can actually be resumed from.

    get_last_checkpoint() only pattern-matches the `checkpoint-<step>` directory NAME, so it happily
    returns a checkpoint whose contents are incomplete and the resume then fails deep inside Trainer.
    This bit us for real: of this run's six checkpoints, only 128000 still had its weights and
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


# Resume from checkpoint when safe. Transformers 5.15+ blocks optimizer-state loading on torch < 2.6.
newest_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
last_checkpoint = find_resumable_checkpoint(training_args.output_dir)
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
        "Checkpoint found but skipping resume because this environment uses torch<2.6; "
        "starting from scratch to avoid blocked optimizer-state loading."
    )
    last_checkpoint = None
else:
    print("No checkpoint found, starting training from scratch")

trainer.train(resume_from_checkpoint=last_checkpoint)

# Standalone export of the finished model — this was missing entirely, so the ONLY artifacts this script
# ever produced were step checkpoints subject to save_total_limit rotation. Weights-only (~270 MB vs the
# ~800 MB checkpoints, which also carry optimizer state), so this is the artifact to hand to NLI
# fine-tuning and the only one worth uploading.
# Deliberately BEFORE the full-test-set eval below: that eval is ~213k forward passes over 3.4M lines,
# and if it gets killed the weights are already safely on disk.
final_dir = mlm_run_dir / 'final'
trainer.save_model(str(final_dir))          # writes config.json + model.safetensors
tokenizer.save_pretrained(str(final_dir))   # writes tokenizer.json + tokenizer_config.json + special_tokens_map.json
print(f"Final model + tokenizer saved to {final_dir}")

# Full-corpus eval for the metric you'd actually report — the per-step evals above only ever saw the 20k subsample
final_metrics = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
final_metrics['perplexity'] = math.exp(final_metrics['eval_loss'])  # the number worth quoting; README target is loss < 2.0 i.e. ppl < 7.4
print(f"Final full test-set metrics: {final_metrics}")

# Persist metrics to the RUN ROOT, not inside a checkpoint. These were previously print-only, so the only
# durable record of any metric was the trainer_state.json that Trainer happens to drop inside each
# checkpoint — which save_total_limit rotates away. Segment 1's numbers survived by luck for exactly that
# reason. Both files here are small JSON and sit outside the .gitignore'd checkpoint-*/final globs, so they
# get committed as the permanent record of the run.
with open(mlm_run_dir / 'final_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(final_metrics, f, indent=2)
trainer.state.save_to_json(str(mlm_run_dir / 'trainer_state.json'))  # full log_history: every logged loss, LR and eval_loss
print(f"Metrics and full log history saved to {mlm_run_dir}")