"""NLI fine-tuning (Step 4) — IndicXNLI Tamil on top of the from-scratch TamBERT MLM backbone.

Follows notebooks/TamBERT_NLI.ipynb, with the checkpoint hygiene from scripts/mlm/train_roberta.py:
validated resume, a rotation-proof metric history, and a standalone weights-only export for the next stage.

This is a CROSS-ENCODER fine-tune (pair -> 3-way label), matching the notebook. It trains the backbone on
Tamil entailment, and the exported encoder is what STS fine-tuning (Step 5) then consumes. Note that the
top-level README's Step 4 describes the SBERT *siamese* recipe (SentenceTransformer + losses.SoftmaxLoss),
which shapes pooled embeddings directly; that is a different objective and would need sentence-transformers.
"""

import json
import math
import os
import random
import zipfile

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from transformers import (DataCollatorWithPadding, EarlyStoppingCallback, PreTrainedTokenizerFast,
                          RobertaForSequenceClassification, Trainer, TrainingArguments)

from paths import Paths
from tokenizer.core.constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN
from tokenizer.core.preprocess import build_text_preprocessor
from training_utils import RunRootStateWriter, export_final_model, resolve_resume_checkpoint, save_run_artifacts

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BACKBONE_RUN = '03_sandhi_codepoint_bert'   # the MLM run whose final/ export we fine-tune
MAX_LENGTH = 128                            # from the notebook; truncation rate is reported below so this stays honest
NLI_GDRIVE_ID = '196RTClD2_Yg7xpXhQ3Mn0FVSD9vBeuBo'

# IndicXNLI/XNLI label convention, confirmed against the notebook's own sampled examples
ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

paths = Paths()

# Load data
extract_dir = paths.nli_data / 'indicxnli_tamil'
split_files = {split: extract_dir / f'{split}.json' for split in ('train', 'dev', 'test')}

if all(p.is_file() for p in split_files.values()):
    print(f"IndicXNLI Tamil already present at {extract_dir}")
else:
    import gdown

    archive = paths.nli_data / 'indicxnli_tamil.zip'
    if not archive.is_file():
        print(f"Downloading IndicXNLI Tamil to {archive}")
        gdown.download(f'https://drive.google.com/uc?id={NLI_GDRIVE_ID}', str(archive), quiet=False)

    print(f"Extracting {archive} to {paths.nli_data}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(paths.nli_data)

    missing = [str(p) for p in split_files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Archive extracted but these splits are missing: {missing}")

_SPLIT_KEYS = {'train': 'train', 'dev': 'validation', 'test': 'test'}  # dev.json stores its rows under 'validation'


def data_to_df(json_path):
    """Load one IndicXNLI split into the notebook's column names.

    encoding='utf-8' is not optional: Windows defaults open() to cp1252, which cannot decode Tamil at all.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data[_SPLIT_KEYS[json_path.stem]])
    return df.rename(columns={'premise': 'sentence1', 'hypothesis': 'sentence2', 'label': 'gold_label'})


train_df = data_to_df(split_files['train'])
dev_df = data_to_df(split_files['dev'])
test_df = data_to_df(split_files['test'])

for name, df in (('train', train_df), ('dev', dev_df), ('test', test_df)):
    counts = df['gold_label'].value_counts().sort_index()
    breakdown = ', '.join(f"{ID2LABEL[label]} {count}" for label, count in counts.items())
    print(f"{name:5s}: {len(df):>7,} rows  ({breakdown})")

# ---------------------------------------------------------------------------------------------------
# Backbone + tokenizer
# ---------------------------------------------------------------------------------------------------
backbone_dir = paths.mlm_run_generator(BACKBONE_RUN) / 'final'
if not backbone_dir.is_dir():
    raise FileNotFoundError(
        f"MLM backbone not found at {backbone_dir}. Finish scripts/mlm/train_roberta.py first (it writes "
        f"that directory at the end of training), or copy the exported final/ folder there."
    )

# Built the same way as in train_roberta.py rather than via AutoTokenizer: the special tokens are stated
# explicitly, so this cannot silently depend on whatever tokenizer_config.json happened to be written.
raw_tokenizer = Tokenizer.from_file(str(backbone_dir / 'tokenizer.json'))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                    cls_token=BOS_TOKEN,
                                    sep_token=EOS_TOKEN,
                                    pad_token=PAD_TOKEN,
                                    unk_token=UNK_TOKEN,
                                    mask_token=MASK_TOKEN)

# BACKBONE_RUN doubles as the tokenizer variant name, so the preprocessing this tokenizer needs is read off that
# variant's own config.yaml instead of hardcoded here. Raises rather than silently no-op'ing if the name is not a
# known variant, since "no preprocessing" is indistinguishable from correct behaviour until the metrics come in.
preprocess_text, preprocessing_label = build_text_preprocessor(BACKBONE_RUN, paths)
print(f"Input preprocessing for {BACKBONE_RUN}: {preprocessing_label}")

# The lm_head is dropped and a fresh 3-way classifier is initialized — the "newly initialized" warning
# transformers prints here is expected, not a problem.
model = RobertaForSequenceClassification.from_pretrained(
    str(backbone_dir),
    num_labels=len(ID2LABEL),
    id2label=ID2LABEL,      # saved into config.json, so the exported model documents its own label order
    label2id=LABEL2ID,
)
print(f"Loaded backbone from {backbone_dir} with a fresh {len(ID2LABEL)}-way classification head")

# ---------------------------------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------------------------------
# The tokenizer's TemplateProcessing pair template produces [CLS] A [SEP] B [SEP] — verified — so the
# premise/hypothesis boundary is marked by a token the MLM already saw.
#
# return_token_type_ids=False is load-bearing. That pair template assigns type_id 1 to sentence B, but the
# MLM config was built with type_vocab_size=1, so token_type_embeddings has exactly ONE row and an id of 1
# is an out-of-bounds embedding lookup ("IndexError: index out of range in self", or an opaque device-side
# assert on GPU). transformers 5.14 happens not to emit token_type_ids for this tokenizer, so the default is
# currently safe by accident; this makes it safe on purpose. RoBERTa is single-token-type by design anyway.
#
# Both sentences go through preprocess_text first. The backbone's tokenizer (03_sandhi_codepoint_bert) learned
# its vocab from sandhi-marked text and the MLM saw nothing else, so feeding raw IndicXNLI here silently wastes
# 1,253 of the 32,000 embedding rows -- every vocab entry containing the ⟂ marker becomes unreachable -- and
# pushes fertility from 1.3412 to 1.4141 (~5% more tokens against the 128-token budget). No error either way,
# which is why this is derived from the variant's own config rather than left to be remembered.
def tokenize_function(batch):
    encoded = tokenizer([preprocess_text(s) for s in batch['sentence1']],
                        [preprocess_text(s) for s in batch['sentence2']],
                        truncation=True,
                        max_length=MAX_LENGTH,
                        return_token_type_ids=False)
    encoded['num_tokens'] = [len(ids) for ids in encoded['input_ids']]
    return encoded


def to_dataset(df, name):
    # preserve_index=False keeps datasets from smuggling in an __index_level_0__ column
    ds = Dataset.from_pandas(df[['sentence1', 'sentence2', 'gold_label']], preserve_index=False)
    ds = ds.map(tokenize_function, batched=True, remove_columns=['sentence1', 'sentence2'])
    ds = ds.rename_column('gold_label', 'labels')

    # No padding above — DataCollatorWithPadding pads per batch instead of padding everything to MAX_LENGTH.
    # Report what MAX_LENGTH actually costs rather than assuming the notebook's 128 is generous. num_tokens
    # rides along as its own column so this stat reads a few MB of ints instead of materializing every
    # input_ids list for all ~393k training rows.
    lengths = np.asarray(ds['num_tokens'])
    truncated = int((lengths >= MAX_LENGTH).sum())
    print(f"{name:5s}: median {int(np.median(lengths))} tokens, p99 {int(np.percentile(lengths, 99))}, "
          f"truncated at {MAX_LENGTH}: {truncated:,} ({truncated / len(ds):.2%})")
    return ds.remove_columns('num_tokens')


train_ds = to_dataset(train_df, 'train')
dev_ds = to_dataset(dev_df, 'dev')
test_ds = to_dataset(test_df, 'test')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    """Accuracy plus macro-F1 and per-class F1. numpy only, to avoid adding a scikit-learn dependency."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1_scores = []
    for label_id in ID2LABEL:
        true_positives = int(((predictions == label_id) & (labels == label_id)).sum())
        false_positives = int(((predictions == label_id) & (labels != label_id)).sum())
        false_negatives = int(((predictions != label_id) & (labels == label_id)).sum())
        denominator = 2 * true_positives + false_positives + false_negatives
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positives / denominator)

    metrics = {'accuracy': float((predictions == labels).mean()),
               'macro_f1': float(np.mean(f1_scores))}
    metrics.update({f'f1_{ID2LABEL[label_id]}': f1_scores[label_id] for label_id in ID2LABEL})
    return metrics


# ---------------------------------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------------------------------
nli_run_dir = paths.nli_run_generator(BACKBONE_RUN)  # resolved once; reused for output_dir and the export below

NUM_EPOCHS = 3          # notebook value, and the standard BERT fine-tuning budget
TRAIN_BATCH_SIZE = 16   # notebook value; no gradient accumulation, so logged loss is NOT inflated as it was in MLM
WARMUP_RATIO = 0.06

# count is what makes a schedule reproducible across a resume (LambdaLR restores last_epoch but rebuilds its
# lambda from whatever is passed here — a changed ratio silently shifts the LR curve mid-run).
steps_per_epoch = math.ceil(len(train_ds) / TRAIN_BATCH_SIZE)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = round(WARMUP_RATIO * total_steps)
print(f"Schedule: {steps_per_epoch:,} steps/epoch x {NUM_EPOCHS} epochs = {total_steps:,} steps, "
      f"warmup {warmup_steps:,} ({WARMUP_RATIO:.0%})")

training_args = TrainingArguments(
    output_dir=str(nli_run_dir),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=32,          # notebook value
    learning_rate=2e-5,                     # fine-tuning LR, 5x lower than the 1e-4 used for MLM pre-training
    lr_scheduler_type='linear',
    warmup_steps=warmup_steps,
    weight_decay=0.01,                      # unlike the MLM run (0): mild regularization on a 393k-row supervised set
    eval_strategy='steps',                  # steps, not epoch: 3 epochs is only 3 eval points, too coarse for early stopping
    eval_steps=2000,                        # ~24.5k steps/epoch, so ~12 evals/epoch. The dev set is 3.2k rows (~102
                                            # forward passes at batch 32), so unlike MLM this needs no subsampling
    save_strategy='steps',
    save_steps=2000,                        # must equal eval_steps for load_best_model_at_end
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',  # accuracy, not eval_loss: the labels are balanced 1:1:1 and accuracy is
    greater_is_better=True,                 # the number this task is judged on
    logging_steps=200,
    fp16=True,                              # GPU only — will fail on a CPU-only torch build
    seed=SEED,
)

# patience is in EVAL CALLS: 5 x eval_steps=2000 is a 10,000-step (~0.4 epoch) no-improvement window
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
run_root_state_writer = RunRootStateWriter(nli_run_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,              
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback, run_root_state_writer],
)

last_checkpoint = resolve_resume_checkpoint(training_args.output_dir)

trainer.train(resume_from_checkpoint=last_checkpoint)

# Weights-only export — the artifact STS fine-tuning consumes
export_final_model(trainer, tokenizer, nli_run_dir)

trainer.remove_callback(early_stopping_callback)

final_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix='test')
print(f"Final IndicXNLI Tamil test metrics: {final_metrics}")
save_run_artifacts(trainer, nli_run_dir, final_metrics)
