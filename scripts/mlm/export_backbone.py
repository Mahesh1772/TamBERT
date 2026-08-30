"""Build mlm/<run>/final/ from an existing checkpoint, without retraining.

train_roberta.py writes final/ only once trainer.train() has returned and the export line is reached, so a
run that was killed leaves its weights inside checkpoint-<step>/ but never produces the artifact NLI
fine-tuning consumes. This produces it after the fact.

WHICH checkpoint is the whole point of this script. With load_best_model_at_end=True the run records its
best model in trainer_state.json as best_model_checkpoint, and that is NOT the highest-numbered directory
whenever early stopping fired: EarlyStoppingCallback(patience=15) at eval_steps=10000 stops a full 150,000
steps AFTER the best eval_loss. Exporting the newest checkpoint would ship a model the run itself measured
as worse, and nothing downstream would ever tell you.

Usage:
    python scripts/mlm/export_backbone.py [run_name] [--from checkpoint-NNNNNN] [--force]

Deliberately reads only weights, never optimizer state, so it has no torch>=2.6 requirement and runs fine
on a CPU-only box.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, RobertaForSequenceClassification

from paths import Paths
from tokenizer.core.constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN
from training_utils import _CHECKPOINT_DIR_RE, _WEIGHT_FILES

DEFAULT_RUN = '03_sandhi_codepoint_bert'

# Not description=__doc__: argparse would print the docstring's em dashes straight to a cp1252 Windows
# console, which mojibakes them. Every print in this script is deliberately ASCII for the same reason.
parser = argparse.ArgumentParser(description='Build mlm/<run>/final/ from an existing checkpoint.')
parser.add_argument('run_name', nargs='?', default=DEFAULT_RUN, help=f'MLM run under mlm/ (default: {DEFAULT_RUN})')
parser.add_argument('--from', dest='source', default=None,
                    help='checkpoint dir name to export, overriding the best/newest choice')
parser.add_argument('--force', action='store_true', help='overwrite an existing final/')
args = parser.parse_args()

paths = Paths()
run_dir = paths.mlm / args.run_name
if not run_dir.is_dir():
    sys.exit(f"No such MLM run: {run_dir}")


def has_weights(checkpoint_dir):
    return any((checkpoint_dir / w).is_file() for w in _WEIGHT_FILES)


# Only weights are needed for an export, so this is deliberately looser than
# training_utils.find_resumable_checkpoint, which additionally demands optimizer.pt for a resume.
checkpoints = sorted(
    ((int(m.group(1)), d) for d in run_dir.iterdir() if d.is_dir() and (m := _CHECKPOINT_DIR_RE.match(d.name))),
    key=lambda pair: pair[0],
)
exportable = [(step, d) for step, d in checkpoints if has_weights(d)]
for step, d in checkpoints:
    if not has_weights(d):
        print(f"Skipping {d.name}: no {' or '.join(_WEIGHT_FILES)}")

# ---------------------------------------------------------------------------------------------------
# What the run itself says about where it landed
# ---------------------------------------------------------------------------------------------------
# The run-root copy (written by RunRootStateWriter) is preferred: it survives save_total_limit rotation,
# whereas each checkpoint's own copy only ever reflects that checkpoint.
state_candidates = [run_dir / 'trainer_state.json'] + [d / 'trainer_state.json' for _, d in reversed(exportable)]
state_path = next((p for p in state_candidates if p.is_file()), None)
state = {}

if state_path is None:
    print("No trainer_state.json found - cannot tell which checkpoint was best; falling back to the newest.")
else:
    with open(state_path, encoding='utf-8') as f:
        state = json.load(f)
    print(f"\nRun state from {state_path.relative_to(paths.root)}")
    print(f"  global_step        : {state.get('global_step'):,}")
    print(f"  epoch              : {state.get('epoch')}")

    best_metric = state.get('best_metric')
    if best_metric is not None:
        print(f"  best eval_loss     : {best_metric:.4f}  (perplexity {math.exp(best_metric):,.1f})")
    print(f"  best_model_checkpoint: {state.get('best_model_checkpoint')}")

    # The tail of the eval curve is what shows whether the stop was a real plateau or a premature cut.
    evals = [e for e in state.get('log_history', []) if 'eval_loss' in e]
    if evals:
        print(f"\n  last {min(8, len(evals))} of {len(evals)} evals (step / eval_loss / ppl):")
        for e in evals[-8:]:
            marker = '  <- best' if best_metric is not None and abs(e['eval_loss'] - best_metric) < 1e-9 else ''
            print(f"    {e['step']:>9,}  {e['eval_loss']:.4f}  {math.exp(e['eval_loss']):>7,.1f}{marker}")

# ---------------------------------------------------------------------------------------------------
# Pick the source checkpoint: explicit override > run's own best > newest with weights
# ---------------------------------------------------------------------------------------------------
if args.source:
    source = run_dir / Path(args.source).name
    if not source.is_dir():
        sys.exit(f"--from {args.source} does not exist under {run_dir}")
    if not has_weights(source):
        sys.exit(f"--from {args.source} has no {' or '.join(_WEIGHT_FILES)}")
    reason = 'explicitly requested via --from'
else:
    if not exportable:
        sys.exit(f"No checkpoint under {run_dir} contains model weights - nothing to export.")

    newest = exportable[-1][1]
    best_recorded = state.get('best_model_checkpoint')
    # best_model_checkpoint is an absolute path from the machine that trained, so match on the basename only.
    best = run_dir / Path(best_recorded).name if best_recorded else None

    if best and best.is_dir() and has_weights(best):
        source = best
        reason = "the run's own best_model_checkpoint"
        if source != newest:
            print(f"\nNOTE: exporting {source.name}, NOT the newest checkpoint {newest.name}. Early stopping "
                  f"means the tail of the run was worse than its best - the newest weights are not the good ones.")
    else:
        source = newest
        if best_recorded:
            print(f"\nWARNING: best_model_checkpoint is {Path(best_recorded).name} but that directory is not "
                  f"present with weights here. Exporting {newest.name} instead - this may be a WORSE model than "
                  f"the run achieved. Copy the best checkpoint over if you still have it.")
            reason = 'newest available; the recorded best is missing'
        else:
            reason = 'newest with weights (no best recorded)'

print(f"\nExporting from {source.name} ({reason})")

final_dir = run_dir / 'final'
if final_dir.exists() and any(final_dir.iterdir()) and not args.force:
    sys.exit(f"{final_dir} already exists and is non-empty. Pass --force to overwrite.")
final_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
# Copy weights + config; regenerate the tokenizer from source
# ---------------------------------------------------------------------------------------------------
for name in ('config.json',) + _WEIGHT_FILES:
    src = source / name
    if src.is_file():
        shutil.copy2(src, final_dir / name)
        print(f"  copied {name} ({src.stat().st_size / 1e6:,.1f} MB)")

# Rebuilt from tokenizer/<run>/tokenizer.json exactly as train_roberta.py does, rather than copying the
# checkpoint's files: this guarantees the special-token map is the one the model was actually trained with.
tokenizer_path = paths.tokenizer_name_generator(args.run_name) / 'tokenizer.json'
if not tokenizer_path.is_file():
    sys.exit(f"Tokenizer not found at {tokenizer_path} - needed to write a self-contained final/")

raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                    cls_token=BOS_TOKEN,
                                    sep_token=EOS_TOKEN,
                                    pad_token=PAD_TOKEN,
                                    unk_token=UNK_TOKEN,
                                    mask_token=MASK_TOKEN)
tokenizer.save_pretrained(str(final_dir))
print(f"  wrote tokenizer from {tokenizer_path.relative_to(paths.root)}")

# ---------------------------------------------------------------------------------------------------
# Verify: this is the exact load train_nli.py performs, so a failure here is a failure there
# ---------------------------------------------------------------------------------------------------
weights = load_file(str(final_dir / 'model.safetensors'))
nan_tensors = [k for k, t in weights.items() if t.isnan().any() or t.isinf().any()]
if nan_tensors:
    sys.exit(f"CORRUPT: {len(nan_tensors)} tensor(s) contain NaN/Inf, e.g. {nan_tensors[:3]}. Do not use this.")

vocab_size = raw_tokenizer.get_vocab_size()
embedding_rows = next(t.shape[0] for k, t in weights.items() if k.endswith('embeddings.word_embeddings.weight'))
if embedding_rows != vocab_size:
    sys.exit(f"MISMATCH: word_embeddings has {embedding_rows:,} rows but the tokenizer vocab is {vocab_size:,}")

# The "newly initialized classifier" warning below is expected — it is the point of a fresh head.
model = RobertaForSequenceClassification.from_pretrained(str(final_dir), num_labels=3)
trainable = sum(p.numel() for p in model.parameters())

print(f"\nVerified: {len(weights)} tensors, no NaN/Inf, vocab {vocab_size:,} matches embeddings")
print(f"Loads as RobertaForSequenceClassification: {trainable / 1e6:,.1f}M params, "
      f"{model.config.num_hidden_layers}L / {model.config.hidden_size}H")
print(f"\nBackbone ready at {final_dir}\nNext: python scripts/nli/train_nli.py")
