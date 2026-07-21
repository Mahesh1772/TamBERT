"""
Shared grapheme-cluster remapping module.

WHY THIS EXISTS
----------------
The `tokenizers` library's `initial_alphabet` parameter silently truncates
any multi-character string down to its first character (confirmed in the
official docs: "If the strings contain more than one character, only the
first one is kept."). Tamil grapheme clusters are mostly multi-codepoint
(base consonant + vowel sign + optional virama), so passing them into
`initial_alphabet` never actually seeds full clusters — it silently
degrades to seeding individual codepoints, which the trainer would have
picked up anyway. This is almost certainly the root cause of the
~1,492-vocab cap seen before.

THE FIX
-------
Remap each distinct multi-codepoint grapheme cluster in the corpus to a
single placeholder codepoint (from the Unicode Private Use Area) before
training. A placeholder is exactly one codepoint, so BPE/Unigram treats
each grapheme cluster as atomic automatically — merges can combine
placeholders together (spanning multiple original graphemes), but can
never land in the middle of one.

Single-codepoint clusters (plain ASCII, isolated Tamil vowels/consonants,
digits, punctuation, the sandhi marker (Sequence[U+27C2] / whatever glyph
you use) are left untouched — there's nothing to protect them from, so
remapping them would just waste placeholder slots.

UNIGRAM-ONLY RELABEL TRICK
---------------------------
`restore_vocab_in_place` rewrites a saved Unigram tokenizer.json's vocab
piece strings, swapping placeholders back for real grapheme text. This
works for Unigram because its decode is a Viterbi search over literal
substrings of the input against literal vocab pieces, not a fixed
per-codepoint initial split + ordered merge application. Once relabelled,
the saved tokenizer works directly on ordinary raw Tamil text — no
runtime substitution step needed downstream.

This does NOT carry over to BPE. BPE's merge rules are applied starting
from a fixed initial per-codepoint split of the raw input; relabelling
the merge rules after training wouldn't make them match a raw-text
codepoint split anymore. If you want to fix the BPE grapheme variant
later, that needs the substitution step kept live at both train AND
inference time (a real, permanent preprocessing step) — don't reuse the
relabel-after-training trick from this module for BPE.
"""
from __future__ import annotations
import json
import regex as re
from pathlib import Path

GRAPHEME_RE = re.compile(r"\X")
PUA_START = 0xE000
PUA_END = 0xF8FF  # 6,400 slots in the Private Use Area — comfortably more than Tamil needs


def find_multi_codepoint_graphemes(*corpus_paths: Path) -> list[str]:
    """Scan one or more corpus files and return the sorted, deduplicated
    list of grapheme clusters that span more than one Unicode codepoint.
    Single-codepoint clusters are excluded on purpose — see module docstring."""
    clusters: set[str] = set()
    for p in corpus_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                clusters.update(GRAPHEME_RE.findall(line))
    clusters.discard(" ")
    return sorted(c for c in clusters if len(c) > 1)


def build_placeholder_map(multi_codepoint_graphemes: list[str]) -> dict[str, str]:
    """Assign each multi-codepoint grapheme cluster a unique single-codepoint
    placeholder from the Unicode Private Use Area."""
    if len(multi_codepoint_graphemes) > (PUA_END - PUA_START + 1):
        raise ValueError(
            f"{len(multi_codepoint_graphemes)} distinct multi-codepoint graphemes "
            f"found — exceeds the {PUA_END - PUA_START + 1} available PUA slots. "
            f"(Worth double-checking that's a real number and not a normalizer bug "
            f"producing spurious clusters before raising the range.)"
        )
    return {g: chr(PUA_START + i) for i, g in enumerate(multi_codepoint_graphemes)}


def save_map(grapheme_to_placeholder: dict[str, str], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(grapheme_to_placeholder, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_map(map_path: Path) -> dict[str, str]:
    return json.loads(map_path.read_text(encoding="utf-8"))


def substitute_line(line: str, grapheme_to_placeholder: dict[str, str]) -> str:
    """Replace every occurrence of a mapped grapheme cluster with its
    placeholder. Clusters not in the map (single-codepoint, or genuinely
    unseen at map-build time) pass through unchanged rather than erroring —
    graceful degradation to plain codepoint handling for anything unmapped."""
    return "".join(grapheme_to_placeholder.get(g, g) for g in GRAPHEME_RE.findall(line))


def substitute_file(src_path: Path, dst_path: Path, grapheme_to_placeholder: dict[str, str]) -> None:
    with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(substitute_line(line, grapheme_to_placeholder))


def restore_text(text: str, placeholder_to_grapheme: dict[str, str]) -> str:
    for placeholder, grapheme in placeholder_to_grapheme.items():
        text = text.replace(placeholder, grapheme)
    return text


def restore_vocab_in_place(tokenizer_json_path: Path, grapheme_to_placeholder: dict[str, str]) -> None:
    """Post-training step, Unigram only: rewrite a saved tokenizer.json's
    vocab piece strings, swapping placeholders back for original grapheme
    text. See module docstring for why this is safe for Unigram specifically.

    NOTE: verify the ["model"]["vocab"] key path against your installed
    `tokenizers` version's actual tokenizer.json output before relying on
    this — print/inspect one saved file first if in doubt. This matches the
    schema as of the versions checked while writing this, but serialization
    details can shift between releases.
    """
    placeholder_to_grapheme = {v: k for k, v in grapheme_to_placeholder.items()}
    data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]  # list of [piece, score] pairs for Unigram
    data["model"]["vocab"] = [
        [restore_text(piece, placeholder_to_grapheme), score] for piece, score in vocab
    ]
    tokenizer_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
