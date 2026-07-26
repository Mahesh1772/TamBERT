from __future__ import annotations
import json
import regex as re
from pathlib import Path
from tqdm import tqdm

GRAPHEME_RE = re.compile(r"\X")
PUA_START = 0xE000
PUA_END = 0xF8FF  # 6,400 slots in the Private Use Area — comfortably more than Tamil needs


def find_multi_codepoint_graphemes(corpus_paths: list[Path], line_counts: list[int]) -> list[str]:
    """Scan one or more corpus files and return the sorted, deduplicated
    list of grapheme clusters that span more than one Unicode codepoint.
    Single-codepoint clusters are excluded on purpose — see module docstring."""
    clusters: set[str] = set()
    for p, line_count in zip(corpus_paths, line_counts):
        with open(p, encoding="utf-8") as f:
            for line in tqdm(f, total=line_count, desc=f"Scanning {p.name}", unit="lines"):
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


def substitute_file(src_path: Path, dst_path: Path, grapheme_to_placeholder: dict[str, str], line_count: int) -> None:
    with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=line_count, desc=f"Substituting {src_path.name}", unit="lines"):
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
