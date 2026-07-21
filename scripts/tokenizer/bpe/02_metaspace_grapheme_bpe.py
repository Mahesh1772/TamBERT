from time import time, process_time
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sys, json
import regex as re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import SAMPLE_TEXT, calculate_tokenizer_metrics, UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths

paths = Paths()

# Define the tokenizer
whitespace_grapheme_bpe = Tokenizer(BPE(unk_token=UNK_TOKEN))

# Normalizer, as previously defined in the `TamBert Data Analysis Notebook` file
whitespace_grapheme_bpe.normalizer = normalizers.Sequence([normalizers.NFC(),
                                                 normalizers.Replace(Regex(r",+"), ","),
                                                 normalizers.Replace(Regex(r"\.+"), "."),
                                                 normalizers.Replace(Regex(r"'+"), "'"),
                                                 normalizers.Replace(Regex(r"-+"), "-")])

# Pre-tokenizer — Metaspace: bakes a word-boundary marker (▁)
# directly into token text, so decoding survives arbitrary subword splitting
whitespace_grapheme_bpe.pre_tokenizer = pre_tokenizers.Metaspace()

# Get a starting pont for the distinct graphemes in the training data, excluding whitespace
print('Getting a starting point for the distinct graphemes in the training data, excluding whitespace...')
grapheme_re = re.compile(r"\X")
distinct_graphemes = set()
with open(paths.train, encoding='utf-8') as f:
    for line in f:
        distinct_graphemes.update(grapheme_re.findall(line))
distinct_graphemes.discard(' ')
distinct_graphemes = sorted(distinct_graphemes)
print(f"Found {len(distinct_graphemes)} distinct graphemes in the training data, excluding whitespace.")

print("Pre-tokenization process:")
print(whitespace_grapheme_bpe.pre_tokenizer.pre_tokenize_str(SAMPLE_TEXT))

# Trainer
whitespace_grapheme_bpe_trainer = BpeTrainer(
    special_tokens=BPE_SPECIAL_TOKENS,
    vocab_size=VOCAB_SIZE,
    initial_alphabet=distinct_graphemes,
)

# Train
start_cpu_time, start_wall_time = process_time(), time()
whitespace_grapheme_bpe.train([str(paths.train)], trainer=whitespace_grapheme_bpe_trainer)
end_cpu_time, end_wall_time = process_time(), time()
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Decoder — must match the pre-tokenizer's marker scheme
whitespace_grapheme_bpe.decoder = decoders.Metaspace()

# Evaluate
train_metrics = calculate_tokenizer_metrics(whitespace_grapheme_bpe, paths.test)

# Post-processor — BERT [CLS]/[SEP] structure, set after training since it
# needs real token IDs from the trained vocab
whitespace_grapheme_bpe.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", whitespace_grapheme_bpe.token_to_id("[CLS]")),
        ("[SEP]", whitespace_grapheme_bpe.token_to_id("[SEP]")),
    ],
)

# Report
print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
print(f"Vocabulary size: {len(whitespace_grapheme_bpe.get_vocab())}")
print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

encoded = whitespace_grapheme_bpe.encode(SAMPLE_TEXT)
print(f"Encoding (with [CLS]/[SEP]): {encoded.tokens}")
print(f"Decoded: {whitespace_grapheme_bpe.decode(encoded.ids)!r}")

# Save
out_dir = paths.tokenizer_name_generator('metaspace_grapheme_bpe')
whitespace_grapheme_bpe.save(str(out_dir / 'tokenizer.json'))


with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(whitespace_grapheme_bpe.get_vocab()),
        'wall_time_seconds': wall_time_taken,
    }, f, indent=2)

print(f"Saved to {out_dir}")