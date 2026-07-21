from time import time, process_time
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sys, json
import regex as re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import calculate_tokenizer_metrics
from constants import UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS, SAMPLE_TEXT
from grapheme_remap import load_map, substitute_line
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths

paths = Paths()

placeholder_map = load_map(paths.grapheme_placeholder_map)

# Define the tokenizer
sandhi_grapheme_bpe = Tokenizer(BPE(unk_token=UNK_TOKEN))

# Normalizer
sandhi_grapheme_bpe.normalizer = normalizers.Sequence([normalizers.NFC(),
                                                 normalizers.Replace(Regex(r",+"), ","),
                                                 normalizers.Replace(Regex(r"\.+"), "."),
                                                 normalizers.Replace(Regex(r"'+"), "'"),
                                                 normalizers.Replace(Regex(r"-+"), "-")])

# Pre-tokenizer — Metaspace: bakes a word-boundary marker (▁)
# directly into token text, so decoding survives arbitrary subword splitting
sandhi_grapheme_bpe.pre_tokenizer = pre_tokenizers.Metaspace()

print("Pre-tokenization process (on a placeholder-substituted sample):")
sample_substituted = substitute_line(SAMPLE_TEXT, placeholder_map)
print(sandhi_grapheme_bpe.pre_tokenizer.pre_tokenize_str(sample_substituted))

print("Pre-tokenization process:")
print(sandhi_grapheme_bpe.pre_tokenizer.pre_tokenize_str(SAMPLE_TEXT))

# Trainer
sandhi_grapheme_bpe_trainer = BpeTrainer(
    special_tokens=BPE_SPECIAL_TOKENS,
    vocab_size=VOCAB_SIZE,
)

# Train
start_cpu_time, start_wall_time = process_time(), time()
sandhi_grapheme_bpe.train([str(paths.train_sandhi_grapheme_marked)], trainer=sandhi_grapheme_bpe_trainer)
end_cpu_time, end_wall_time = process_time(), time()
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Decoder — must match the pre-tokenizer's marker scheme
sandhi_grapheme_bpe.decoder = decoders.Sequence([decoders.Metaspace(),
                                        decoders.Replace(Regex("⟂"), "")])

# Evaluate
train_metrics = calculate_tokenizer_metrics(sandhi_grapheme_bpe, paths.test_sandhi_grapheme_marked)

# Post-processor — BERT [CLS]/[SEP] structure, set after training since it
# needs real token IDs from the trained vocab
sandhi_grapheme_bpe.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", sandhi_grapheme_bpe.token_to_id("[CLS]")),
        ("[SEP]", sandhi_grapheme_bpe.token_to_id("[SEP]")),
    ],
)

# Report
print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
print(f"Vocabulary size: {len(sandhi_grapheme_bpe.get_vocab())}")
print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

encoded = sandhi_grapheme_bpe.encode(SAMPLE_TEXT)
print(f"Encoding (with [CLS]/[SEP]): {encoded.tokens}")
print(f"Decoded: {sandhi_grapheme_bpe.decode(encoded.ids)!r}")

# Save
out_dir = paths.tokenizer_name_generator('sandhi_grapheme_bpe')
sandhi_grapheme_bpe.save(str(out_dir / 'tokenizer.json'))


with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(sandhi_grapheme_bpe.get_vocab()),
        'wall_time_seconds': wall_time_taken,
    }, f, indent=2)

print(f"Saved to {out_dir}")