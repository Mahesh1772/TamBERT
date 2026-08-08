from tokenizers import Tokenizer
from transformers import RobertaConfig, EarlyStoppingCallback, RobertaForMaskedLM, DataCollatorForLanguageModeling, PreTrainedTokenizerFast, Trainer, TrainingArguments
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from paths import Paths
from tokenizer.core.constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN
from datasets import load_dataset

paths = Paths()

tokenizer_path = paths.tokenizer_name_generator('03_sandhi_codepoint_bert') / 'tokenizer.json'  # confirmed lowest fertility (1.341) of all 12 variants

# Single load, reused for config sizing below and for encode_batch further down (was loaded twice before)
raw_tokenizer = Tokenizer.from_file(tokenizer_path)

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
    eval_strategy='epoch',                  # evaluate once per epoch
    save_strategy='epoch',                  # checkpoint once per epoch, aligned with eval for load_best_model_at_end
    num_train_epochs=6,                     # ceiling, not a target; early stopping + load_best_model_at_end guard against overshooting
    learning_rate=1e-4,                     # standard from-scratch RoBERTa/BERT peak LR
    lr_scheduler_type='linear',             # linear decay after warmup; matches original RoBERTa recipe
    warmup_ratio=0.06,                      # ~6% of steps ramp-up; restored, was missing here — important for a randomly-initialized transformer
    weight_decay=0,                         # no weight decay; consider ~0.01 later if eval loss plateaus/overfits
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,          # effective batch size = 16*4 = 64
    fp16=True,                              # mixed precision for GPU throughput/memory headroom
    group_by_length=True,                   # NEW: bucket similar-length sequences together; big win given how skewed your token-length histogram is toward short lines
    save_total_limit=2,                     # keep 2 checkpoints: room for the current "best" plus the latest, so load_best_model_at_end never gets evicted mid-run
    load_best_model_at_end=True,            # always finish with the lowest eval_loss checkpoint, not just the last epoch
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

# Callbacks
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# DataLoading
raw_tokenizer.enable_truncation(max_length=512)  # required: your token-length histogram has outlier lines up to ~2500 tokens, which would exceed max_position_embeddings without this

def tokenize_function(examples):
    encoding = raw_tokenizer.encode_batch(examples['text'])
    return {'input_ids': [e.ids for e in encoding],
            'attention_mask': [e.attention_mask for e in encoding]}

tokenized_datasets = data.map(tokenize_function, batched=True, remove_columns=['text'])

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    callbacks=[early_stopping_callback]  # was defined but never attached before — early stopping wasn't actually active
)

trainer.train()