from time import time, process_time
from tokenizers import Regex, Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import calculate_tokenizer_metrics
from constants import UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS, SAMPLE_TEXT, standard_normalizer
from grapheme_remap import load_map, substitute_line, restore_text
from sandhi import sandhi_mark_boundaries
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths

paths = Paths()

placeholder_map = load_map(paths.grapheme_placeholder_map)
placeholder_to_grapheme = {v: k for k, v in placeholder_map.items()}

# Define the tokenizer
sandhi_grapheme_bpe = Tokenizer(BPE(unk_token=UNK_TOKEN))

# Normalizer
sandhi_grapheme_bpe.normalizer = standard_normalizer

# Pre-tokenizer — Metaspace: bakes a word-boundary marker (▁)
# directly into token text, so decoding survives arbitrary subword splitting
sandhi_grapheme_bpe.pre_tokenizer = pre_tokenizers.Metaspace()

sample_sandhi_marked = sandhi_mark_boundaries(SAMPLE_TEXT, lang="ta")
sample_substituted = substitute_line(sample_sandhi_marked, placeholder_map)

print("Pre-tokenization process (on a sandhi-marked, placeholder-substituted sample):")
print(sandhi_grapheme_bpe.pre_tokenizer.pre_tokenize_str(sample_substituted))

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

encoded = sandhi_grapheme_bpe.encode(sample_substituted)
print(f"Encoding on placeholder text (with [CLS]/[SEP]): {encoded.tokens}")

decoded_placeholder = sandhi_grapheme_bpe.decode(encoded.ids)
print(f"Decoded (restored to raw Tamil): {restore_text(decoded_placeholder, placeholder_to_grapheme)!r}")

# Save
out_dir = paths.tokenizer_name_generator('04_sandhi_grapheme_bpe')
sandhi_grapheme_bpe.save(str(out_dir / 'tokenizer.json'))


with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(sandhi_grapheme_bpe.get_vocab()),
        'wall_time_seconds': wall_time_taken,
        'cpu_time_seconds': cpu_time_taken,
        'sample_text': {
            'raw': SAMPLE_TEXT,
            'sandhi_marked': sample_sandhi_marked,
            'placeholder_substituted': sample_substituted,
            'encoded_tokens': encoded.tokens,
            'decoded_placeholder': decoded_placeholder,
            'decoded_restored': restore_text(decoded_placeholder, placeholder_to_grapheme)
        }
    }, f, indent=2, ensure_ascii=False)

print(f"Saved to {out_dir}")