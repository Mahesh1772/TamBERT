from tokenizer.core.grapheme_remap import find_multi_codepoint_graphemes, build_placeholder_map, save_map, substitute_file
from paths import Paths

paths = Paths()

TRAIN_LINES = 30_683_869
TEST_LINES = 3_409_318

# Build the map from the plain (non-sandhi) train + test files. Sandhi
# marking only inserts the boundary marker between existing grapheme
# clusters — it doesn't change what those clusters are — so one shared
# map correctly covers both the plain and sandhi variants. Scanning both
# train and test here (not just train) so nothing in test silently falls
# back to unmapped/degraded handling purely because it didn't happen to
# occur in train — this is just enumerating the character alphabet, not
# learning from labels, so it doesn't leak anything statistical.
print("Scanning for multi-codepoint grapheme clusters...")
multi_graphemes = find_multi_codepoint_graphemes([paths.train, paths.test], [TRAIN_LINES, TEST_LINES])
print(f"Found {len(multi_graphemes)} distinct multi-codepoint grapheme clusters.")

placeholder_map = build_placeholder_map(multi_graphemes)
map_path = paths.grapheme_placeholder_map
save_map(placeholder_map, map_path)
print(f"Saved placeholder map ({len(placeholder_map)} entries) to {map_path}")

# Plain grapheme variant — feeds 03_train_unigram_grapheme.py
substitute_file(paths.train, paths.train_grapheme_marked, placeholder_map, TRAIN_LINES)
substitute_file(paths.test, paths.test_grapheme_marked, placeholder_map, TEST_LINES)
print(f"Wrote {paths.train_grapheme_marked} and {paths.test_grapheme_marked}")

# Sandhi + grapheme variant — feeds 05_train_unigram_sandhi_grapheme.py
substitute_file(paths.train_sandhi_marked, paths.train_sandhi_grapheme_marked, placeholder_map, TRAIN_LINES)
substitute_file(paths.test_sandhi_marked, paths.test_sandhi_grapheme_marked, placeholder_map, TEST_LINES)
print(f"Wrote {paths.train_sandhi_grapheme_marked} and {paths.test_sandhi_grapheme_marked}")
