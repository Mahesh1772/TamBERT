from time import time, process_time
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import calculate_tokenizer_metrics
from constants import UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS, SAMPLE_TEXT
from grapheme_remap import load_map, restore_vocab_in_place, substitute_line
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths

paths = Paths()

# Same shared map as 03 — sandhi marking doesn't change what the grapheme
# clusters are, just where the ⟂ boundary markers sit between them.
placeholder_map = load_map(paths.grapheme_placeholder_map)

# Define the tokenizer
grapheme_unigram_sandhi = Tokenizer(Unigram())

# Normalizer — unchanged from the working reference
grapheme_unigram_sandhi.normalizer = normalizers.Sequence([normalizers.NFC(),
                                                   normalizers.Replace(Regex(r",+"), ","),
                                                   normalizers.Replace(Regex(r"\.+"), "."),
                                                 normalizers.Replace(Regex(r"'+"), "'"),
                                                 normalizers.Replace(Regex(r"-+"), "-")])

# Pre-tokenizer — plain Metaspace. Same reasoning as 03: grapheme atomicity
# comes from the placeholder-substituted data, not initial_alphabet.
grapheme_unigram_sandhi.pre_tokenizer = pre_tokenizers.Metaspace()

print("Pre-tokenization process (on a placeholder-substituted sample):")
sample_substituted = substitute_line(SAMPLE_TEXT, placeholder_map)
print(grapheme_unigram_sandhi.pre_tokenizer.pre_tokenize_str(sample_substituted))

# Trainer — no initial_alphabet
grapheme_unigram_sandhi_trainer = UnigramTrainer(special_tokens=BPE_SPECIAL_TOKENS,
                                                  vocab_size=VOCAB_SIZE,
                                                  unk_token=UNK_TOKEN)

# Train on the sandhi+grapheme-marked file (from grapheme_precompute.py,
# built on top of paths.train_sandhi_marked)
start_cpu_time, start_wall_time = process_time(), time()
grapheme_unigram_sandhi.train([str(paths.train_sandhi_grapheme_marked)], trainer=grapheme_unigram_sandhi_trainer)
end_cpu_time, end_wall_time = process_time(), time()
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Decoder — Metaspace + strip the sandhi marker, same as the working
# sandhi reference files. Text is still in placeholder form at this point.
grapheme_unigram_sandhi.decoder = decoders.Sequence([decoders.Metaspace(),
                                        decoders.Replace(Regex("⟂"), "")])

# Evaluate on the sandhi+grapheme-marked test file
train_metrics = calculate_tokenizer_metrics(grapheme_unigram_sandhi, paths.test_sandhi_grapheme_marked)

# Post-processor — BERT [CLS]/[SEP] structure
grapheme_unigram_sandhi.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", grapheme_unigram_sandhi.token_to_id("[CLS]")),
        ("[SEP]", grapheme_unigram_sandhi.token_to_id("[SEP]")),
    ],
)

# Report
print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
print(f"Vocabulary size: {len(grapheme_unigram_sandhi.get_vocab())}")
print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

encoded = grapheme_unigram_sandhi.encode(sample_substituted)
print(f"Encoding on placeholder text (with [CLS]/[SEP]): {encoded.tokens}")
print(f"Decoded: {grapheme_unigram_sandhi.decode(encoded.ids)!r}")

# Save
out_dir = paths.tokenizer_name_generator('04_sandhi_grapheme_unigram')
grapheme_unigram_sandhi.save(str(out_dir / 'tokenizer.json'))

# Relabel placeholders back to real Tamil grapheme text — same Unigram-only
# trick as 03. Note this variant's vocab still legitimately contains the
# sandhi marker (⟂) as its own token where the model learned to use it —
# that's expected and correct, only the placeholder codepoints get swapped.
restore_vocab_in_place(out_dir / 'tokenizer.json', placeholder_map)

# Sanity check — reload and confirm it works on raw text. Caveat: this
# variant was trained on sandhi-marked input, so for a fully faithful
# check you'd run SAMPLE_TEXT through your sandhi-marking step first
# (sandhi.py wasn't available here to call directly) — this check only
# confirms the placeholder relabel itself worked, not the sandhi marking.
reloaded = Tokenizer.from_file(str(out_dir / 'tokenizer.json'))
raw_encoded = reloaded.encode(SAMPLE_TEXT)
print(f"Sanity check — encoding raw (non-sandhi-marked) text after relabelling: {raw_encoded.tokens}")

with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(grapheme_unigram_sandhi.get_vocab()),
        'wall_time_seconds': wall_time_taken,
    }, f, indent=2)

print(f"Saved to {out_dir}")
