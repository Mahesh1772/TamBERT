# TamilBERT MLM Pre-training — 03_sandhi_codepoint_bert_6L768H

## Full run contents (checkpoints, logs, trainer_state.json)
> **Model Artifacts:** [Google Drive Folder](https://drive.google.com/drive/folders/1-fFuIsgYCrKbmJWZA-sdtO_UPyk6Fi1W?usp=sharing)

## Run identity
- Tokenizer: `03_sandhi_codepoint_bert` (fertility 1.341 — lowest of the 12 variants evaluated)
- Architecture: `RobertaForMaskedLM`, 6 hidden layers, 12 attention heads, hidden_size 768 (DistilRoBERTa-sized), vocab_size 32000
- `max_position_embeddings`: 516 (512 + pad_token_id offset + 1, to cover RoBERTa's padding-index position scheme)

## Training configuration
- Corpus: 30,683,869 train lines / 3,409,318 test lines (sandhi-marked Tamil)
- Effective batch size: 64 (`per_device_train_batch_size=16` × `gradient_accumulation_steps=4`)
- LR schedule: linear, peak 1e-4, `warmup_ratio=0.06` (note: actual observed warmup in this run completed by ~step 2,500, well short of the intended ~172,800 steps — see run log for detail)
- `num_train_epochs=6` (ceiling; `EarlyStoppingCallback` patience=15 eval calls)
- Checkpointing/eval every 2000 steps on a 20k-line shuffled subsample of the test set; full test-set eval run once at the end of training
- Interruptions: this run was interrupted and resumed at least twice, including a mid-run switch from CPU to GPU

## Results
- Final eval_loss / perplexity: [FILL IN]
- Final step reached / early-stopped vs. completed all 6 epochs: [FILL IN]
- Best checkpoint: [FILL IN]
- Qualitative masked-fill sanity check: [FILL IN — pass/notes]

## Next step
NLI fine-tuning (planned; not yet started as of this run's completion)