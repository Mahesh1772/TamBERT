from time import time, process_time
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import calculate_tokenizer_metrics
from constants import UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS, SAMPLE_TEXT
from sandhi import sandhi_mark_boundaries
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths

paths = Paths()

# Define the tokenizer
unigram_sandhi = Tokenizer(Unigram())

# Normalizer, as previously defined in the `TamBert Data Analysis Notebook` file
unigram_sandhi.normalizer = normalizers.Sequence([normalizers.NFC(),
                                                   normalizers.Replace(Regex(r",+"), ","),
                                                   normalizers.Replace(Regex(r"\.+"), "."),
                                                 normalizers.Replace(Regex(r"'+"), "'"),
                                                 normalizers.Replace(Regex(r"-+"), "-")])

# Pre-tokenizer — Metaspace: bakes a word-boundary marker (▁)
# directly into token text, so decoding survives arbitrary subword splitting
unigram_sandhi.pre_tokenizer = pre_tokenizers.Metaspace()

sample_marked = sandhi_mark_boundaries(SAMPLE_TEXT, lang="ta")
print("Pre-tokenization process (on a sandhi-marked sample):")
print(unigram_sandhi.pre_tokenizer.pre_tokenize_str(sample_marked))

# Trainer
unigram_sandhi_trainer = UnigramTrainer(special_tokens=BPE_SPECIAL_TOKENS,
                                         vocab_size=VOCAB_SIZE,
                                         unk_token=UNK_TOKEN)

# Train
start_cpu_time, start_wall_time = process_time(), time()
unigram_sandhi.train([str(paths.train_sandhi_marked)], trainer=unigram_sandhi_trainer)
end_cpu_time, end_wall_time = process_time(), time()
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Decoder — must match the pre-tokenizer's marker scheme
unigram_sandhi.decoder = decoders.Sequence([decoders.Metaspace(),
                                        decoders.Replace(Regex("⟂"), "")])

# Evaluate
train_metrics = calculate_tokenizer_metrics(unigram_sandhi, paths.test_sandhi_marked)

# Post-processor — BERT [CLS]/[SEP] structure, set after training since it
# needs real token IDs from the trained vocab
unigram_sandhi.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", unigram_sandhi.token_to_id("[CLS]")),
        ("[SEP]", unigram_sandhi.token_to_id("[SEP]")),
    ],
)

# Report
print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
print(f"Vocabulary size: {len(unigram_sandhi.get_vocab())}")
print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

encoded = unigram_sandhi.encode(sample_marked)
print(f"Encoding (with [CLS]/[SEP]): {encoded.tokens}")
print(f"Decoded: {unigram_sandhi.decode(encoded.ids)!r}")

# Save
out_dir = paths.tokenizer_name_generator('03_sandhi_codepoint_unigram')
unigram_sandhi.save(str(out_dir / 'tokenizer.json'))


with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(unigram_sandhi.get_vocab()),
        'wall_time_seconds': wall_time_taken,
    }, f, indent=2)

print(f"Saved to {out_dir}")