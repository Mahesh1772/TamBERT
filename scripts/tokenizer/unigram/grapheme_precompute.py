"""
Offline precompute step for grapheme-atomic tokenizer training — same role
in the pipeline as sandhi_precompute.py, just for grapheme clusters instead
of sandhi boundaries. Run this once before 03_train_unigram_grapheme.py or
05_train_unigram_sandhi_grapheme.py.

REQUIRES: your Paths class needs four new attributes for this to run —
    train_grapheme_marked, test_grapheme_marked,
    train_sandhi_grapheme_marked, test_sandhi_grapheme_marked
mirroring however train_sandhi_marked / test_sandhi_marked are already
defined there. Also assumes a `paths.tokenizer_dir` (or equivalent) to
save the placeholder map next to — swap in whatever attribute you actually
use for that if the name differs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grapheme_remap import (
    find_multi_codepoint_graphemes,
    build_placeholder_map,
    save_map,
    substitute_file,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import Paths

paths = Paths()

# Build the map from the plain (non-sandhi) train + test files. Sandhi
# marking only inserts the boundary marker between existing grapheme
# clusters — it doesn't change what those clusters are — so one shared
# map correctly covers both the plain and sandhi variants. Scanning both
# train and test here (not just train) so nothing in test silently falls
# back to unmapped/degraded handling purely because it didn't happen to
# occur in train — this is just enumerating the character alphabet, not
# learning from labels, so it doesn't leak anything statistical.
print("Scanning for multi-codepoint grapheme clusters...")
multi_graphemes = find_multi_codepoint_graphemes(paths.train, paths.test)
print(f"Found {len(multi_graphemes)} distinct multi-codepoint grapheme clusters.")

placeholder_map = build_placeholder_map(multi_graphemes)
map_path = paths.tokenizer_dir / "grapheme_placeholder_map.json"
save_map(placeholder_map, map_path)
print(f"Saved placeholder map ({len(placeholder_map)} entries) to {map_path}")

# Plain grapheme variant — feeds 03_train_unigram_grapheme.py
substitute_file(paths.train, paths.train_grapheme_marked, placeholder_map)
substitute_file(paths.test, paths.test_grapheme_marked, placeholder_map)
print(f"Wrote {paths.train_grapheme_marked} and {paths.test_grapheme_marked}")

# Sandhi + grapheme variant — feeds 05_train_unigram_sandhi_grapheme.py
substitute_file(paths.train_sandhi_marked, paths.train_sandhi_grapheme_marked, placeholder_map)
substitute_file(paths.test_sandhi_marked, paths.test_sandhi_grapheme_marked, placeholder_map)
print(f"Wrote {paths.train_sandhi_grapheme_marked} and {paths.test_sandhi_grapheme_marked}")
