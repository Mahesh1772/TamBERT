from tokenizers import Tokenizer
from transformers import RobertaConfig, EarlyStoppingCallback, RobertaForMaskedLM, DataCollatorForLanguageModeling, PreTrainedTokenizerFast, Trainer, TrainingArguments
import os
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
training_args = TrainingArguments(
    output_dir=str(paths.mlm_run_generator('03_sandhi_codepoint_bert')),  # named per tokenizer + architecture, avoids checkpoint collisions across runs
    eval_strategy='steps',                  # CHANGED from 'epoch': must match save_strategy for load_best_model_at_end (needed by EarlyStoppingCallback) — see eval_subset below for why this is cheap
    eval_steps=2000,                        # matches save_steps so every save has a fresh eval_loss to compare against
    save_strategy='steps',                  # CHANGED from 'epoch': one epoch is ~479k steps here, far too coarse for crash recovery
    save_steps=2000,                        # NEW: checkpoint roughly every 2000 steps (~240 checkpoints/epoch); tune up once you've seen the I/O overhead per save
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
# patience is in EVAL CALLS, not epochs — at eval_steps=2000 that's a ~30,000-step no-improvement window (was
# effectively ~1.4M steps at patience=3 epochs before); loosen/tighten once you've seen how noisy eval_loss is step-to-step
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=15)

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
    callbacks=[early_stopping_callback]  # was defined but never attached before — early stopping wasn't actually active
)

# Resume from checkpoint when safe. Transformers 5.15+ blocks optimizer-state loading on torch < 2.6.
last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
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

# Full-corpus eval for the metric you'd actually report — the per-step evals above only ever saw the 20k subsample
final_metrics = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
print(f"Final full test-set metrics: {final_metrics}")