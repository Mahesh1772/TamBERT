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

Six checkpoints exist (128000, 130000, 132000, 518000, 546000, 548000). Originally only **checkpoint-128000**
was complete: checkpoint-130000 had weights but no `optimizer.pt`, and the other four had neither. Every
surviving file in the gutted checkpoints was small (`scheduler.pt` ~1 KB, `rng_state.pth` ~14 KB) and both
missing files were the large ones (~270 MB weights, ~540 MB optimizer state), so this was a transfer/sync
failure, not a Trainer bug — `_save_checkpoint` writes the model *before* `scheduler.pt`, so the weights did
exist on disk.

`best_model_checkpoint` in the recovered state points at
`C:\Users\butte\OneDrive\Documents\GitHub\TamBERT\mlm\...`. **The run trained inside a OneDrive-synced
folder**, which is the most likely cause of the loss: OneDrive dehydrates or fails to sync large files.
Consequence for any future run: write `output_dir` to a **non-synced local path**.

### ✅ checkpoint-548000 recovered (2026-08-30)

Re-uploaded complete from the origin machine — all ten files present, and the sizes confirm the large ones
are whole rather than truncated:

| File | Size | Expected |
|---|---|---|
| `model.safetensors` | 259.9 MB | ~259 MB for 6L/768H/32k-vocab |
| `optimizer.pt` | 519.8 MB | 2× weights (Adam m + v) ✅ |
| `trainer_state.json` | 266 KB | full `log_history` to step 548,000 |

This makes 548,000 resumable (`find_resumable_checkpoint` needs weights + `optimizer.pt`; both are there) and
means **steps 128,000–548,000 are no longer lost** — that 266 KB `trainer_state.json` holds the whole curve.
Those intermediate numbers are not yet transcribed into this README.

⚠️ 548,000 is the run's **stopping point, not its best model** — see below. It is the right thing to resume
*from*, and the wrong thing to fine-tune *on*.

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

**Full test-set eval: not recorded for this segment.** The script only runs it after `trainer.train()`
returns. It *did* eventually run, at the end of the whole run — see segment 2 below.

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

## Results — final, full test set

The run did finish: **early stopping fired at step 548,000** and the end-of-training full-corpus eval ran.
These numbers are transcribed from stdout into [final_metrics.json](final_metrics.json) — the Aug 20 script
(commit `280b486`) printed them but had no persistence step, so stdout was the only copy.

| | |
|---|---|
| Full test-set eval_loss | **3.7678** (perplexity **43.3**) |
| Measured on | the **full** 3,409,318-line test set (3,413,207 samples in 3.84 h), not the 20k subsample |
| Weights measured | **`checkpoint-518000`** — `load_best_model_at_end=True` swapped the best checkpoint in before this eval |
| Step reached | 548,000 = 1.1417 epochs = **19.03%** of the 6-epoch ceiling |
| Stopping reason | Early stopping. `patience=15` × `eval_steps=2000` = a 30,000-step no-improvement window, so best = 548,000 − 30,000 = **518,000**, corroborated by `checkpoint-518000` surviving rotation |
| vs. segment 1 | −0.5879 nats from 4.3558; perplexity 77.9 → 43.3 (not strictly comparable: subsample vs. full test set) |

**Distance to target:** [README.md](../../README.md) wants MLM loss < 2.0 (perplexity 7.4), so **1.7679 nats
to go**.

### The plateau is at high LR, not a capacity ceiling

Early stopping at 19% of the schedule means the linear decay never annealed — LR was still ~8.1e-5 of its
1e-4 peak when the run stopped. A large share of MLM's final quality comes from decaying LR toward zero, so
"stopped improving" here means *stopped improving at 8.1e-5*, not *out of headroom*. The cheap experiment is
to resume from `checkpoint-548000` with `num_train_epochs` reduced to ~1.5 so the decay actually completes
over the remaining ~170,000 steps, rather than training longer against the original 2.88M-step ramp.

⚠️ **No `final/` was ever written.** The Aug 20 script had no weights-only export (added in `1e9d075`), so
the only place these weights exist is inside the surviving `checkpoint-*` directories. Use
`scripts/mlm/export_backbone.py` to build `final/` from a checkpoint after the fact — it reads
`best_model_checkpoint` out of `trainer_state.json` and defaults to the **best** step, not the newest.

⚠️ **But checkpoint-518000 is still gutted — only 548000 was recovered.** So the model that produced the
3.7678 above is *not currently available*, and the export has to fall back to 548,000 (the script warns loudly
when it does). The two are 30,000 steps of measured non-improvement apart, comfortably inside the ±0.03
masking noise floor, so 548,000 is a fine backbone — just do not quote 3.7678 as its score. Worth asking the
origin machine for `checkpoint-518000/model.safetensors` (259.9 MB; the 519.8 MB `optimizer.pt` is *not*
needed for a backbone export, only for resuming).

## Next step
NLI fine-tuning — `scripts/nli/train_nli.py` (IndicXNLI Tamil, cross-encoder). It consumes
`mlm/03_sandhi_codepoint_bert/final/`, which must be produced by `export_backbone.py` first since this run
never wrote it. Expect modest accuracy from a perplexity-43 6-layer encoder; the immediate value is the first
real downstream signal plus an end-to-end check of the pipeline before STS.
