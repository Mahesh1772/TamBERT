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

# Load the grapheme -> placeholder map built by grapheme_precompute.py.
# Run that script first if this errors on a missing file.
placeholder_map = load_map(paths.grapheme_placeholder_map)

# Define the tokenizer
grapheme_unigram = Tokenizer(Unigram())

# Normalizer — unchanged from the working reference
grapheme_unigram.normalizer = normalizers.Sequence([normalizers.NFC(),
                                                    normalizers.Replace(Regex(r",+"), ","),
                                                    normalizers.Replace(Regex(r"\.+"), "."),
                                                 normalizers.Replace(Regex(r"'+"), "'"),
                                                 normalizers.Replace(Regex(r"-+"), "-")])

# Pre-tokenizer — plain Metaspace. Grapheme atomicity now comes from the
# placeholder-substituted training data itself (grapheme_precompute.py),
# not from initial_alphabet — that parameter silently truncates any
# multi-character entry to one codepoint, which is almost certainly what
# capped the vocab at ~1,492 last time. No fix to that parameter exists;
# this sidesteps it entirely instead.
grapheme_unigram.pre_tokenizer = pre_tokenizers.Metaspace()

print("Pre-tokenization process (on a placeholder-substituted sample):")
sample_substituted = substitute_line(SAMPLE_TEXT, placeholder_map)
print(grapheme_unigram.pre_tokenizer.pre_tokenize_str(sample_substituted))

# Trainer — no initial_alphabet needed; each placeholder codepoint is
# already atomic by construction, so the trainer can't split inside one.
grapheme_unigram_trainer = UnigramTrainer(special_tokens=BPE_SPECIAL_TOKENS,
                                           vocab_size=VOCAB_SIZE,
                                           unk_token=UNK_TOKEN)

# Train on the placeholder-substituted file (from grapheme_precompute.py)
start_cpu_time, start_wall_time = process_time(), time()
grapheme_unigram.train([str(paths.train_grapheme_marked)], trainer=grapheme_unigram_trainer)
end_cpu_time, end_wall_time = process_time(), time()
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Decoder — plain Metaspace. Text is still in placeholder form at this
# point; the vocab gets relabelled back to real Tamil text further down.
grapheme_unigram.decoder = decoders.Metaspace()

# Evaluate on the placeholder-substituted test file — fair apples-to-apples
# comparison against what the model was actually trained on.
train_metrics = calculate_tokenizer_metrics(grapheme_unigram, paths.test_grapheme_marked)

# Post-processor — BERT [CLS]/[SEP] structure
grapheme_unigram.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", grapheme_unigram.token_to_id("[CLS]")),
        ("[SEP]", grapheme_unigram.token_to_id("[SEP]")),
    ],
)

# Report
print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
print(f"Vocabulary size: {len(grapheme_unigram.get_vocab())}")
print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

encoded = grapheme_unigram.encode(sample_substituted)
print(f"Encoding on placeholder text (with [CLS]/[SEP]): {encoded.tokens}")
print(f"Decoded: {grapheme_unigram.decode(encoded.ids)!r}")

# Save
out_dir = paths.tokenizer_name_generator('02_metaspace_grapheme_unigram')
grapheme_unigram.save(str(out_dir / 'tokenizer.json'))

# Relabel the saved vocab from placeholders back to real Tamil grapheme
# text — safe for Unigram (see grapheme_remap.py docstring for why). After
# this, the saved tokenizer.json works on ordinary raw Tamil text directly,
# no runtime placeholder substitution needed downstream.
restore_vocab_in_place(out_dir / 'tokenizer.json', placeholder_map)

# Sanity check — actually verify the claim above rather than trust it:
# reload the relabelled tokenizer and confirm it tokenizes ordinary RAW
# Tamil text (the original SAMPLE_TEXT, not the placeholder version).
reloaded = Tokenizer.from_file(str(out_dir / 'tokenizer.json'))
raw_encoded = reloaded.encode(SAMPLE_TEXT)
print(f"Sanity check — encoding raw (non-substituted) text after relabelling: {raw_encoded.tokens}")

with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(grapheme_unigram.get_vocab()),
        'wall_time_seconds': wall_time_taken,
    }, f, indent=2)

print(f"Saved to {out_dir}")
