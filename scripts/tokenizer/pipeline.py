"""
Shared tokenizer training pipeline for the metaspace/sandhi x
codepoint/grapheme ablation grid, reused by every model family (BPE,
Unigram, WordPiece/BERT). This module is the only place that calls into
the `tokenizers` API; each family's own train_{model}_tokenizer.py is a
thin entry point that points this at that family's config.yaml.

What's declared per family in config.yaml, instead of branched on by
name in code, so adding a new family or changing an assumption never
requires touching this file:

  model_type
      "bpe" | "unigram" | "wordpiece" — selects the Model + Trainer pair
      (see _MODEL_BUILDERS) and, for "wordpiece", whether the decoder
      needs the "##" prefix-stripping step.

  relabel_vocab_after_save
      For grapheme variants only: relabel the saved vocab's placeholder
      codepoints back to real Tamil graphemes in-place, so the saved
      tokenizer.json works on raw text with no runtime substitution
      needed downstream. Verified safe for Unigram (no merge list to
      keep in sync). NOT verified safe for BPE/WordPiece, whose merges
      reference vocab entries — leave False there unless checked.
"""
from time import time, process_time
from pathlib import Path
import sys, json

import yaml
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

# pipeline.py lives alongside metrics.py/constants.py/etc.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import calculate_tokenizer_metrics
from constants import UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS, SAMPLE_TEXT, standard_normalizer
from grapheme_remap import load_map, substitute_line, restore_text, restore_vocab_in_place
from sandhi import sandhi_mark_boundaries


_MODEL_BUILDERS = {
    "bpe": lambda: (
        BPE(unk_token=UNK_TOKEN),
        BpeTrainer(special_tokens=BPE_SPECIAL_TOKENS, vocab_size=VOCAB_SIZE),
    ),
    "unigram": lambda: (
        Unigram(),
        UnigramTrainer(special_tokens=BPE_SPECIAL_TOKENS, vocab_size=VOCAB_SIZE, unk_token=UNK_TOKEN),
    ),
    "wordpiece": lambda: (
        WordPiece(),
        WordPieceTrainer(special_tokens=BPE_SPECIAL_TOKENS, vocab_size=VOCAB_SIZE, unk_token=UNK_TOKEN),
    ),
}


def _build_model_and_trainer(model_type):
    try:
        builder = _MODEL_BUILDERS[model_type]
    except KeyError:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {list(_MODEL_BUILDERS)}")
    return builder()


def _preprocess(text, use_sandhi, use_grapheme, placeholder_map):
    """Apply the same preprocessing chain used for the training corpus to a
    single string, so the printed/encoded sample matches what the model
    actually saw. Order matters: sandhi marking happens before grapheme
    substitution."""
    if use_sandhi:
        text = sandhi_mark_boundaries(text, lang="ta")
    if use_grapheme:
        text = substitute_line(text, placeholder_map)
    return text


def _build_decoder(use_sandhi, model_type):
    """WordPiece needs its '##' prefix stripped before Metaspace; sandhi
    marking needs the ⟂ boundary symbol stripped after."""
    steps = []
    if model_type == "wordpiece":
        steps.append(decoders.WordPiece(prefix="##", cleanup=True))
    steps.append(decoders.Metaspace())
    if use_sandhi:
        steps.append(decoders.Replace(Regex("⟂"), ""))
    return steps[0] if len(steps) == 1 else decoders.Sequence(steps)


def _pretokenize_label(use_sandhi, use_grapheme):
    bits = []
    if use_sandhi:
        bits.append("sandhi-marked")
    if use_grapheme:
        bits.append("placeholder-substituted")
    return f" (on a {', '.join(bits)} sample)" if bits else ""


def train_variant(cfg, paths):
    """
    Train, evaluate, and save one tokenizer variant.

    cfg: dict merging a family's top-level config.yaml settings
         (model_type, relabel_vocab_after_save) with one entry from that
         family's `variants:` list (name, use_sandhi, use_grapheme,
         train_attr, test_attr). Use load_config()/run_variants() below
         rather than building this by hand.
    paths: a shared Paths() instance

    Returns the metrics dict that also gets written to <out_dir>/metrics.json.
    """
    name = cfg["name"]
    model_type = cfg["model_type"]
    use_sandhi = cfg.get("use_sandhi", False)
    use_grapheme = cfg.get("use_grapheme", False)
    relabel_vocab_after_save = cfg.get("relabel_vocab_after_save", False)
    train_path = getattr(paths, cfg["train_attr"])
    test_path = getattr(paths, cfg["test_attr"])

    placeholder_map, placeholder_to_grapheme = {}, {}
    if use_grapheme:
        placeholder_map = load_map(paths.grapheme_placeholder_map)
        placeholder_to_grapheme = {v: k for k, v in placeholder_map.items()}

    print(f"\n=== Training variant: {name} ({model_type}) ===")

    # Define the tokenizer
    model, trainer = _build_model_and_trainer(model_type)
    tokenizer = Tokenizer(model)

    # Normalizer, as previously defined in the `TamBert Data Analysis Notebook` file
    tokenizer.normalizer = standard_normalizer

    # Pre-tokenizer — Metaspace: bakes a word-boundary marker (▁) directly
    # into token text, so decoding survives arbitrary subword splitting
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    sample_input = _preprocess(SAMPLE_TEXT, use_sandhi, use_grapheme, placeholder_map)
    print(f"Pre-tokenization process{_pretokenize_label(use_sandhi, use_grapheme)}:")
    print(tokenizer.pre_tokenizer.pre_tokenize_str(sample_input))

    # Train
    start_cpu, start_wall = process_time(), time()
    tokenizer.train([str(train_path)], trainer=trainer)
    end_cpu, end_wall = process_time(), time()
    cpu_time_taken = end_cpu - start_cpu
    wall_time_taken = end_wall - start_wall

    # Decoder — must match the pre-tokenizer / preprocessing marker scheme
    tokenizer.decoder = _build_decoder(use_sandhi, model_type)

    # Evaluate
    train_metrics = calculate_tokenizer_metrics(tokenizer, test_path)

    # Post-processor — BERT [CLS]/[SEP] structure, set after training since
    # it needs real token IDs from the trained vocab
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Report
    print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
    print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

    encoded = tokenizer.encode(sample_input)
    encoding_label = "Encoding on placeholder text" if use_grapheme else "Encoding"
    print(f"{encoding_label} (with [CLS]/[SEP]): {encoded.tokens}")

    decoded = tokenizer.decode(encoded.ids)
    decoded_final = restore_text(decoded, placeholder_to_grapheme) if use_grapheme else decoded
    decoded_label = "Decoded (restored to raw Tamil)" if use_grapheme else "Decoded"
    print(f"{decoded_label}: {decoded_final!r}")

    # Save
    out_dir = paths.tokenizer_name_generator(name)
    tokenizer.save(str(out_dir / 'tokenizer.json'))

    sanity_check_tokens = None
    if use_grapheme and relabel_vocab_after_save:
        # Relabel the saved vocab's placeholder codepoints back to real
        # Tamil grapheme text, in place. After this the saved
        # tokenizer.json works directly on raw Tamil text — no runtime
        # placeholder substitution needed downstream.
        restore_vocab_in_place(out_dir / 'tokenizer.json', placeholder_map)

        # Sanity check — reload and confirm it tokenizes raw text correctly
        # after relabelling, rather than just trusting that it worked.
        reloaded = Tokenizer.from_file(str(out_dir / 'tokenizer.json'))
        raw_text = sandhi_mark_boundaries(SAMPLE_TEXT, lang="ta") if use_sandhi else SAMPLE_TEXT
        raw_encoded = reloaded.encode(raw_text)
        sanity_check_tokens = raw_encoded.tokens
        raw_label = "sandhi-marked raw" if use_sandhi else "raw (non-substituted)"
        print(f"Sanity check — encoding {raw_label} text after relabelling: {sanity_check_tokens}")

    # Build the sample_text report with exactly the fields the original
    # per-variant script for this combination would have written.
    sample_report = {'raw': SAMPLE_TEXT}
    if use_sandhi:
        sample_report['sandhi_marked'] = sandhi_mark_boundaries(SAMPLE_TEXT, lang="ta")
    if use_grapheme:
        sample_report['placeholder_substituted'] = sample_input
    sample_report['encoded_tokens'] = encoded.tokens
    if use_grapheme:
        sample_report['decoded_placeholder'] = decoded
        sample_report['decoded_restored'] = decoded_final
    else:
        sample_report['decoded'] = decoded_final
    if sanity_check_tokens is not None:
        sample_report['sanity_check_encoded_tokens'] = sanity_check_tokens

    metrics_out = {
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(tokenizer.get_vocab()),
        'wall_time_seconds': wall_time_taken,
        'cpu_time_seconds': cpu_time_taken,
        'sample_text': sample_report,
    }

    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_dir}")
    return metrics_out


def load_config(config_path):
    """Read a family's config.yaml and merge its top-level settings
    (model_type, relabel_vocab_after_save, ...) into each variant dict,
    so train_variant() only ever has to deal with one flat dict. A
    variant can override a family-level default by setting its own key."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    family_defaults = {k: v for k, v in cfg.items() if k != 'variants'}
    return [{**family_defaults, **variant} for variant in cfg['variants']]


def run_variants(config_path, paths, only=None):
    """Train every variant in a family's config.yaml (or just `only`, by
    name). Returns {variant_name: metrics_dict}."""
    variants = load_config(config_path)
    if only:
        variants = [v for v in variants if v['name'] == only]
        if not variants:
            raise ValueError(f"No variant named {only!r} in {config_path}")
    return {cfg['name']: train_variant(cfg, paths) for cfg in variants}
