# TamilBERT v1.0 — The MLM Training Run

The previous entry covered what `train_roberta.py` is configured to do. This one covers what happened when it
ran, which is a different and considerably messier story. The run produced a usable model, and it also lost
420,000 steps of metric history to a cause that had nothing to do with machine learning. Most of the checkpoint
plumbing now in `scripts/training_utils.py` exists because of this run.

The authoritative record is `mlm/03_sandhi_codepoint_bert/README.md`, kept alongside the run itself. This entry
is the narrative version.

---

## The result

| | |
|---|---|
| Full test-set `eval_loss` | **3.7678** (perplexity **43.3**) |
| Measured on | the full 3,409,318-line test set, not the 20k subsample |
| Step reached | 548,000 = 1.14 epochs = **19.03%** of the 6-epoch ceiling |
| Stopping reason | Early stopping |
| Target from the README | loss < 2.0 (perplexity 7.4) |
| Distance to target | **1.7679 nats short** |

So the model works, and it is nowhere near the goal. Perplexity 43 means that at each masked position the model
is effectively choosing between about 43 plausible Tamil tokens. That's a genuine language model — random
guessing over a 32,000 vocabulary would be perplexity 32,000 — but it's a long way from perplexity 7.

The interesting part is *why* it's short, because the obvious reading is wrong.

---

## The plateau is a learning-rate artifact, not a capacity ceiling

Early stopping fired, which naively suggests the model had extracted everything it could and a bigger model is
needed. The step count says otherwise.

The run stopped at step 548,000 out of a scheduled 2,879,892. The learning rate schedule is linear decay from a
1e-4 peak toward zero across the whole 2.88M steps, so at step 548,000 the LR was still around **8.1e-5** —
roughly 81% of peak. The decay had barely begun.

This matters more than it sounds. A large share of a language model's final quality comes from the *end* of
training, when the LR has annealed low enough for the weights to settle into a sharp minimum instead of bouncing
around a broad one. Stopping at 19% of the schedule means the model never got that phase at all. "Stopped
improving" here means *stopped improving at a learning rate of 8.1e-5* — which is a completely different claim
from *out of headroom*.

The cheap experiment, not yet run: resume from `checkpoint-548000` with `num_train_epochs` reduced to about 1.5,
so the linear decay actually completes over the remaining ~170,000 steps rather than being stretched across the
original 2.88M-step ramp. That tests whether the plateau dissolves once the LR anneals, and it costs a fraction
of what training longer against the original schedule would.

---

## The checkpoint disaster

Six checkpoints existed at the end of the run: 128000, 130000, 132000, 518000, 546000, 548000. Of those, exactly
**one** was complete.

| Checkpoint | Weights | Optimizer state |
|---|---|---|
| 128000 | present | present |
| 130000 | present | **missing** |
| 132000 | **missing** | **missing** |
| 518000 | **missing** | **missing** |
| 546000 | **missing** | **missing** |
| 548000 | **missing** | **missing** |

### Working out what caused it

The pattern in the surviving files is what identified the cause. In every gutted checkpoint, the files that
survived were the small ones — `scheduler.pt` at about 1 KB, `rng_state.pth` at about 14 KB — and the files that
vanished were the two large ones: roughly 270 MB of weights and 540 MB of optimizer state.

That rules out a bug in `Trainer`. `Trainer._save_checkpoint` writes the model *before* it writes
`scheduler.pt`, so the presence of `scheduler.pt` proves the weights were written to disk successfully. Something
removed them afterwards, and whatever it was selected on file size.

The recovered `trainer_state.json` supplied the answer. Its `best_model_checkpoint` field pointed at:

```
C:\Users\butte\OneDrive\Documents\GitHub\TamBERT\mlm\...
```

**The run trained inside a OneDrive-synced folder.** Cloud sync clients routinely dehydrate large files —
replacing local content with a placeholder that fetches on demand — and fail outright on files that change as
fast as a checkpoint being overwritten. A 540 MB optimizer file rewritten every few thousand steps is close to a
worst case for sync.

This is now the first entry under "load-bearing details" in the project's `CLAUDE.md`: **never train into a
cloud-synced directory.** `output_dir` must be a local path. It is not a machine learning lesson at all, which
is rather the point.

### What was recovered

`checkpoint-548000` was later re-uploaded complete from the origin machine, and the file sizes confirm the large
files are whole rather than truncated:

| File | Size | Expected |
|---|---|---|
| `model.safetensors` | 259.9 MB | ~259 MB for 6L/768H/32k vocab |
| `optimizer.pt` | 519.8 MB | 2× weights, for Adam's two moment estimates |
| `trainer_state.json` | 266 KB | the full log history to step 548,000 |

That 266 KB JSON file is the important one. It holds the entire loss curve, so steps 128,000 to 548,000 are no
longer a black hole in the record.

`checkpoint-518000` — the best model — is **still gutted**, and that has consequences covered further down.

---

## What got built in response

Three pieces of `scripts/training_utils.py` exist because of this run, and all three are now shared by the MLM
and NLI stages.

### `find_resumable_checkpoint()` — don't trust a directory name

HuggingFace ships `get_last_checkpoint()`, which finds the checkpoint to resume from. It works by pattern-matching
the directory **name** for `checkpoint-<number>` and returning the highest. It never looks inside.

Pointed at this run, `get_last_checkpoint()` would have confidently returned `checkpoint-548000` — at the time,
an empty shell — and the resume would have failed somewhere deep inside `Trainer` with an error message about
whatever file it happened to reach for first.

The replacement requires both weights **and** `optimizer.pt` to be present before it will consider a checkpoint,
walks them in order, and returns the highest that qualifies.

Requiring the optimizer file is the subtle half. Resuming from a weights-only checkpoint doesn't fail — it
*succeeds*, and silently reinitializes Adam's moment estimates and restarts the LR schedule from step 0. Adam's
moments encode an accumulated estimate of gradient scale per parameter; throwing them away means the first few
thousand steps after the resume take badly-scaled updates. Combined with an LR schedule that has jumped back to
its warmup phase, the model gets actively degraded. There is no error message, no warning, and nothing in the
loss curve that unambiguously identifies it after the fact. A quiet quality regression is worse than a crash,
because a crash gets fixed.

### `resolve_resume_checkpoint()` — be loud about skipping

A wrapper that adds the two things every caller needs. First, when the newest checkpoint on disk had to be
skipped, it says so explicitly:

```
NOTE: newest checkpoint on disk is checkpoint-548000 but it is not resumable;
falling back to checkpoint-128000
```

Silence there would hide 420,000 steps of data loss behind a run that appeared to start normally.

Second, it gates on torch ≥ 2.6. Transformers 5.15 and later refuse to load optimizer state on older torch
versions, because `torch.load` on an untrusted pickle is a code-execution risk. On an older torch the function
declines to resume and starts fresh, explaining why — rather than attempting a resume that would be silently
downgraded to weights-only, which is precisely the failure mode above.

### `RunRootStateWriter` — keep the metrics out of the blast radius

`Trainer` writes `trainer_state.json` only *inside* each checkpoint directory. Combined with
`save_total_limit=3`, that means rotation deletes the metric history along with the weights, and an interrupted
run leaves nothing at the run root at all.

This run came within one deleted directory of losing its only surviving numbers. The segment-1 metrics exist
purely because `checkpoint-128000` happened to be the checkpoint that kept its copy.

`RunRootStateWriter` is a `TrainerCallback` that mirrors `trainer_state.json` to the run root on every save,
where nothing rotates it away. The cost is negligible — the file is overwritten in place, so it occupies about
0.7 MB at the end of a 2.88M-step run no matter how many times it's written, against 800 MB per checkpoint.

---

## Two numbers that looked like bugs and weren't

Both of these cost real time to chase down, and both are worth recognizing on sight.

### The logged training loss is inflated 4×

`log_history` shows the training loss going from 38.88 at step 500 to 17.87 at step 128,000. Against an
`eval_loss` of 4.36 at the same step, that is not merely high — it's impossible. A cross-entropy loss of 38 over
a 32,000 vocabulary would mean the model was assigning astronomically low probability to the correct answer,
far worse than random guessing.

It's the `gradient_accumulation_steps=4` logging artifact from the previous entry. HuggingFace sums the loss
across the four accumulated micro-batches instead of averaging. The measured ratio is 4.102, close enough to
4 that the remainder is dropout and the trailing-average window. Dividing by 4:

| Step | Logged / 4 | Sanity check |
|---|---|---|
| 500 | 9.72 | `ln(32000) = 10.37` for a random-init 32k-vocab model — correct for a model that has barely started |
| 128,000 | 4.47 | `eval_loss` was 4.3558. Training loss slightly above eval loss is expected: dropout is active during training and the logged value is a trailing 500-step average |

Both land where they should. The rule: **use `eval_loss` as the real metric, and either ignore the `loss` column
or divide it by 4.**

### Warmup was 2,002 steps, not 172,800

`warmup_ratio=0.06` on a 2,879,892-step schedule should give about 172,800 warmup steps. Back-calculating from
the logged learning rates — `step × peak / lr` — gives 2004, 2002, 2001, 2002 at steps 500, 1000, 1500 and 2000.
So segment 1 actually ran with roughly `warmup_steps=2000`, a ratio of 0.0007.

The decay half was correct: its slope of 3.47e-11 per step spans 2,877,892 steps, matching the intended
`max_steps` of 2,879,892. Only the ramp was short. It did no visible harm either — the loss fell cleanly and
`grad_norm` stayed in a healthy 20–35 band throughout.

The trap is in what happens on **resume**. `LambdaLR` restores its step counter from `scheduler.pt`, but it
rebuilds the LR function itself from whatever the code passes at construction time. So resuming this run with the
current `warmup_ratio=0.06` doesn't continue the old schedule — it builds a different one and evaluates it at
the restored step. Concretely, resuming at step 128,000 would drop the LR from 9.56e-5 down to 7.41e-5, ramp it
back up to 1e-4 by step 172,800, and only then start decaying.

To continue the original schedule seamlessly, set `warmup_steps=2000` explicitly, which takes precedence over
`warmup_ratio`. This generalizes: **a learning rate schedule is only reproducible across a resume if it's
specified in absolute steps.** `train_nli.py` was written with this in mind — it computes `warmup_steps` from a
ratio once and passes the resulting count, rather than passing the ratio.

---

## The results, in two segments

The run was interrupted and resumed at least twice, including a mid-run switch from CPU to GPU. Its history
splits into two recoverable pieces.

### Segment 1, from `checkpoint-128000`

| | |
|---|---|
| Best / final `eval_loss` | 4.3558 (perplexity 77.9) |
| Measured on | the 20k shuffled subsample, **not** the full test set |
| Step reached | 128,000 = 0.267 epochs = 4.44% of the schedule |
| `eval_loss` trend | monotonic from 8.3826 at step 2k to 4.3558 at step 128k, ±0.03 noise |
| Stopping reason | neither early-stopped nor completed — the segment was simply cut off |

The trend line matters more than the endpoint. The loss was still falling at about −0.145 per 32,000 steps when
the segment ended, and the early-stopping patience counter was at 0, meaning every single evaluation had been an
improvement. Nothing about this segment suggests a model running out of room.

### Final, on the full test set

| | |
|---|---|
| Full test-set `eval_loss` | 3.7678 (perplexity 43.3) |
| Measured on | the full 3,409,318-line test set — 3,413,207 samples in 3.84 hours |
| Weights measured | **`checkpoint-518000`**, not the final step |
| Step reached | 548,000 = 19.03% of the ceiling |

The stopping point being 548,000 while the *measured* weights are from 518,000 is not an inconsistency. It's
`load_best_model_at_end=True` doing its job: it swapped the best checkpoint back in before the final evaluation
ran. And 518,000 is exactly where the best checkpoint should be — patience of 15 evaluations at
`eval_steps=2000` is a 30,000-step window, so 548,000 − 30,000 = 518,000. That `checkpoint-518000` survived
rotation at all corroborates it, since `load_best_model_at_end` shields the best checkpoint from
`save_total_limit`.

These numbers were transcribed by hand into `final_metrics.json` from the run's stdout, because the August 20
version of the script printed them and had no persistence step. `save_run_artifacts()` exists so that never
happens again, and the JSON file carries a `_provenance` block recording that its contents were typed in rather
than written by code — including the derivations for the stop step and the best step, so the reasoning can be
checked rather than taken on trust.

---

## Recovering the backbone: `export_backbone.py`

The next stage needs `mlm/03_sandhi_codepoint_bert/final/`. That directory **does not exist**, because the
August 20 script had no weights-only export step — it was added afterwards. The only place these weights exist
is inside the surviving `checkpoint-*` directories.

`scripts/mlm/export_backbone.py` builds `final/` from a checkpoint after the fact. The entire point of the
script is *which* checkpoint, and that is a genuinely easy thing to get wrong.

### Newest is not best

With `load_best_model_at_end=True`, the run records its best model in `trainer_state.json` as
`best_model_checkpoint`. Whenever early stopping fires, that is **not** the highest-numbered directory — it's
one full patience window earlier. Exporting the newest checkpoint would ship weights the run itself measured as
worse, and nothing downstream would ever reveal it. The model would just be quietly a bit worse than it should
be, forever.

So the script prefers the run's own `best_model_checkpoint`, falls back to the newest only when the best isn't
available, and prints loudly in either case. It reads `best_model_checkpoint` by basename only, because the
recorded value is an absolute path from the machine that trained — the OneDrive path above.

It also prefers the run-root `trainer_state.json` over any checkpoint's copy, for the reason
`RunRootStateWriter` exists: the run-root copy is the one rotation can't delete.

### The verification step

Before declaring success the script performs the exact load that `train_nli.py` will perform, so a failure
surfaces here rather than at the start of the next stage:

- Every tensor is checked for NaN and Inf. A partially-synced 260 MB safetensors file can deserialize
  perfectly well and be full of garbage.
- `word_embeddings` row count is compared against the tokenizer's vocabulary size. A mismatch means the
  tokenizer and the weights are from different runs, which would otherwise show up as nonsense predictions.
- The model is loaded as `RobertaForSequenceClassification` — the NLI class — and the parameter count printed.

The tokenizer in `final/` is rebuilt from `tokenizer/<run>/tokenizer.json` rather than copied out of the
checkpoint, so the special-token map is guaranteed to be the one the model trained with.

Every print in the script is deliberately plain ASCII, and `argparse` is given a hand-written description rather
than the module docstring. Both are Windows console concessions: the default cp1252 encoding mojibakes em
dashes, which is the same class of problem as the `encoding='utf-8'` rule that runs through this whole project.

### The blocker as things stand

`checkpoint-518000` — the checkpoint that produced the 3.7678 — is still gutted. Only 548,000 was recovered. So
the export has to fall back to 548,000, and the script warns loudly when it does.

Practically this is fine. The two checkpoints are 30,000 steps of *measured non-improvement* apart, and that
window sits comfortably inside the ±0.03 noise floor of masked-token evaluation. 548,000 is a perfectly good
backbone.

What is not fine is quoting 3.7678 as its score. That number belongs to weights that are not currently
available. Worth asking the origin machine for `checkpoint-518000/model.safetensors` — 259.9 MB, and the
519.8 MB `optimizer.pt` is not needed for an export, only for a resume.

---

## What this stage actually taught

The machine learning content of this run is thin: a 6-layer encoder reached perplexity 43 on Tamil and stopped
early at 19% of its schedule for reasons that look like a learning rate schedule rather than a capacity limit.

Everything else that was learned is operational, and most of it is now enforced in code:

- Don't train into a cloud-synced folder.
- Validate a checkpoint's contents, not its name, before resuming.
- A weights-only resume is worse than no resume, because it succeeds.
- Keep the metric history somewhere `save_total_limit` can't reach.
- Persist metrics to disk the moment they're computed; stdout is not storage.
- Specify LR schedules in absolute steps if a resume is ever possible.
- With gradient accumulation, the logged loss is inflated by the accumulation factor.
- After early stopping, the newest checkpoint is not the best one.

The corpus stage's lesson was to check assumptions rather than trusting them. This stage's is narrower and
sharper: the parts of a training run most likely to destroy work aren't the parts involving any machine
learning.

Next: [10_nli_finetuning.md](10_nli_finetuning.md), the first stage that produces a number comparable against
the leaderboard.
