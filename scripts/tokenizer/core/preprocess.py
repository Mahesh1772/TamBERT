"""Runtime text preprocessing for a trained tokenizer variant — the inference-side half of the ablation grid.

Every variant except `01_metaspace_codepoint_*` was trained on a REWRITTEN corpus: sandhi variants on text with
`⟂` inserted at phonological boundaries, grapheme variants on text with multi-codepoint clusters swapped for
Private Use Area placeholders. The tokenizer was never taught to do that rewriting itself (a custom Python
PreTokenizer was too slow — see journal/03_tokenizer_basics.md), so any downstream stage that feeds it raw Tamil
gets a silently degraded tokenization: no error, just worse numbers.

Measured on `03_sandhi_codepoint_bert`, the variant the MLM backbone was trained with:

    input            fertility   marker-bearing vocab entries reached
    sandhi-marked      1.3412     1,207 of 1,253
    raw (unmarked)     1.4141             0 of 1,253

So 1,253 of 32,000 embedding rows (3.9%) become unreachable and sequences run ~5% longer. `train_nli.py`
originally fed raw IndicXNLI text and hit exactly this. This module exists so no stage has to remember the rule
— it asks for the variant by name and gets the right transform.

The markings and their order are the same ones the training corpus was built with; `pipeline.py` delegates here
so there is exactly one implementation and it cannot drift from what the tokenizers were trained on.
"""
from pathlib import Path

from tokenizer.core.grapheme_remap import load_map, substitute_line
from tokenizer.core.sandhi import sandhi_mark_boundaries

# Each model family's config.yaml, resolved from scripts/tokenizer/. Note the .parent.parent: this module lives
# in scripts/tokenizer/core/ but the configs sit one level up beside it, in scripts/tokenizer/<family>/.
_FAMILIES = ('bpe', 'unigram', 'bert')

# (use_sandhi, use_grapheme) -> the Paths attribute holding the corpus in that representation.
_CORPUS_ATTRS = {
    (False, False): '{split}',
    (True, False): '{split}_sandhi_marked',
    (False, True): '{split}_grapheme_marked',
    (True, True): '{split}_sandhi_grapheme_marked',
}


def family_config_paths():
    """Absolute path to every family's config.yaml, so callers don't each hardcode the layout."""
    return [Path(__file__).resolve().parent.parent / family / 'config.yaml' for family in _FAMILIES]


def preprocess_text(text, use_sandhi, use_grapheme, placeholder_map=None):
    """Apply the corpus-building preprocessing chain to one string.

    Order is load-bearing: sandhi marking runs BEFORE grapheme substitution. Reversed, the sandhi regexes would
    be matching Tamil phonological rules against text whose clusters are already opaque placeholder codepoints,
    and would silently match almost nothing.
    """
    if use_sandhi:
        text = sandhi_mark_boundaries(text, lang='ta')
    if use_grapheme:
        text = substitute_line(text, placeholder_map)
    return text


def saved_tokenizer_markings(cfg):
    """The (use_sandhi, use_grapheme) a SAVED tokenizer.json expects on its input.

    Not simply the config's own flags. When `relabel_vocab_after_save` is set, the saved vocab's placeholder
    codepoints were rewritten back to real Tamil, so the tokenizer now wants UNsubstituted text — passing it
    placeholders would miss every entry. That is enabled for Unigram only; BPE and WordPiece merge lists
    reference vocab entries and were never verified safe to relabel, so their grapheme variants still expect
    placeholder input (93% of their vocab entries contain one).
    """
    use_sandhi = cfg.get('use_sandhi', False)
    use_grapheme = cfg.get('use_grapheme', False)
    if use_grapheme and cfg.get('relabel_vocab_after_save', False):
        use_grapheme = False
    return use_sandhi, use_grapheme


def corpus_attr(use_sandhi, use_grapheme, split='test'):
    """Paths attribute naming the corpus file already in the given representation."""
    return _CORPUS_ATTRS[(use_sandhi, use_grapheme)].format(split=split)


def find_variant_config(variant_name):
    """The merged config dict for one variant, searched across all families.

    Imported lazily: pipeline.py imports this module, so a module-level import of load_config would be circular.
    """
    from tokenizer.core.pipeline import load_config

    for config_path in family_config_paths():
        for cfg in load_config(config_path):
            if cfg['name'] == variant_name:
                return cfg

    raise ValueError(
        f"No tokenizer variant named {variant_name!r} in any of "
        f"{[str(p) for p in family_config_paths()]}. Preprocessing cannot be inferred, and guessing wrong is a "
        f"silent quality loss rather than an error, so this refuses to continue."
    )


def build_text_preprocessor(variant_name, paths):
    """Return (preprocess_fn, description) for the named tokenizer variant.

    `preprocess_fn` maps one raw Tamil string to the representation that variant's saved tokenizer expects. For
    a variant needing nothing (`01_metaspace_codepoint_*`) it is the identity function, so callers can apply it
    unconditionally. `description` is a short human-readable label worth printing into a training log, since
    "which representation was this run actually fed" is otherwise invisible after the fact.
    """
    cfg = find_variant_config(variant_name)
    use_sandhi, use_grapheme = saved_tokenizer_markings(cfg)

    # Only touch the placeholder map when it's actually needed — sandhi-only variants shouldn't fail because a
    # grapheme artifact is missing.
    placeholder_map = load_map(paths.grapheme_placeholder_map) if use_grapheme else None

    if not (use_sandhi or use_grapheme):
        return (lambda text: text), 'raw text (no preprocessing)'

    applied = ', '.join(filter(None, ['sandhi marking' if use_sandhi else '',
                                      'grapheme placeholder substitution' if use_grapheme else '']))

    def preprocess_fn(text):
        return preprocess_text(text, use_sandhi, use_grapheme, placeholder_map)

    return preprocess_fn, applied
