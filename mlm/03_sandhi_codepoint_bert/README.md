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
- Schedule length: 479,982 steps/epoch → `max_steps` 2,879,892 for the 6-epoch ceiling
- LR schedule: linear, peak 1e-4
- `num_train_epochs=6` (ceiling; `EarlyStoppingCallback` patience=15 eval calls)
- Checkpointing/eval every 2000 steps on a 20k-line shuffled subsample of the test set; full test-set eval run once at the end of training
- Interruptions: this run was interrupted and resumed at least twice, including a mid-run switch from CPU to GPU

## Artifact loss — read this before trusting the Drive folder

Six checkpoints exist (128000, 130000, 132000, 518000, 546000, 548000) but only **checkpoint-128000** is
complete. checkpoint-130000 has weights but no `optimizer.pt`; the other four have neither. Every surviving
file in the gutted checkpoints is small (`scheduler.pt` ~1 KB, `rng_state.pth` ~14 KB) and both missing files
are the large ones (~270 MB weights, ~540 MB optimizer state), so this was a transfer/sync failure, not a
Trainer bug — `_save_checkpoint` writes the model *before* `scheduler.pt`, so the weights did exist on disk.

`best_model_checkpoint` in the recovered state points at
`C:\Users\butte\OneDrive\Documents\GitHub\TamBERT\mlm\...`. **The run trained inside a OneDrive-synced
folder**, which is the most likely cause of the loss: OneDrive dehydrates or fails to sync large files.
Two consequences:
- The weights may still be recoverable from OneDrive version history / the origin machine — worth checking before writing off 548,000.
- The re-run must write `output_dir` to a **non-synced local path**.

## Results — segment 1 (recovered from checkpoint-128000/trainer_state.json)

Only this segment has recoverable metrics. The run reached at least step 548,000 (the checkpoint directory
exists) but no `trainer_state.json` survives for it, so steps 128,000–548,000 are undocumented and unrecoverable.

| | |
|---|---|
| Best / final eval_loss | **4.3558** (perplexity **77.9**) |
| Measured on | the 20k shuffled test subsample, **not** the full 3.4M-line test set |
| Step reached | 128,000 = 0.267 epochs = **4.44%** of the 6-epoch schedule |
| Best checkpoint | `checkpoint-128000` — i.e. the *last* eval was the best |
| Stopping reason | Neither. Not early-stopped (`early_stopping_patience_counter: 0`), not completed; the segment was cut off |
| eval_loss trend | Monotonic 8.3826 (step 2k) → 4.3558 (step 128k), ±0.03 noise; still falling −0.145 per 32k steps at the cut |
| `total_flos` | 1.129e17 |

**Full test-set eval: never recorded.** The script only runs it after `trainer.train()` returns, which never
happened on an interruption.

### Logged training loss is inflated 4× — not a divergence

`log_history` shows `loss` going 38.88 (step 500) → 17.87 (step 128k), which looks impossible against
`eval_loss` 4.36. It is a `gradient_accumulation_steps=4` logging artifact — the ratio is 4.102. Divided by 4:

| | logged / 4 | sanity check |
|---|---|---|
| step 500 | 9.72 | ln(32000) = 10.37 for a random-init 32k-vocab model ✅ |
| step 128,000 | 4.47 | `eval_loss` 4.3558 (train slightly above eval: dropout on, trailing 500-step average) ✅ |

Both land exactly where they should. **Use `eval_loss` as the real metric; ignore the `loss` column, or divide
it by 4.**

### Warmup was ~2,002 steps, not ~172,800

Confirmed from the logged LRs (`step*peak/lr` gives 2004 / 2002 / 2001 / 2002 at steps 500/1000/1500/2000).
The *decay* was correct — its slope of 3.47e-11/step spans 2,877,892 steps, matching `max_steps` 2,879,892.
So only the ramp was short: this segment ran with roughly `warmup_steps=2000` (ratio 0.0007), not
`warmup_ratio=0.06`. It did no visible harm — loss fell cleanly and `grad_norm` stayed in the 20–35 band.

⚠️ **Resume implication:** the current `warmup_ratio=0.06` builds a *different* schedule. `LambdaLR` restores
`last_epoch` from `scheduler.pt` but takes its lambda from the new construction, so resuming at step 128,000
drops LR from 9.56e-5 to 7.41e-5, ramps back to 1e-4 by step 172,800, and only then decays. To continue
segment 1's schedule seamlessly instead, set `warmup_steps=2000` (it takes precedence over `warmup_ratio`).

## Results — segment 2 / re-run (pending)

Resuming from `checkpoint-128000` costs 420,000 steps ≈ 0.88 epochs ≈ 4.3× the training that survives.

- Final full test-set eval_loss / perplexity: [PENDING]
- Final step reached / early-stopped vs. completed all 6 epochs: [PENDING]
- Best checkpoint: [PENDING]
- Qualitative masked-fill sanity check: [PENDING]

Distance to target: [README.md](../../README.md) wants MLM loss < 2.0 (perplexity 7.4). Segment 1 ended at
4.3558, so **2.36 nats to go** — the remaining 95% of the 6-epoch schedule is what has to close that.

## Next step
NLI fine-tuning (planned; not yet started). It should consume `mlm/03_sandhi_codepoint_bert/final/`, the
weights-only export added to `scripts/mlm/train_roberta.py` — not a rotating step checkpoint.
