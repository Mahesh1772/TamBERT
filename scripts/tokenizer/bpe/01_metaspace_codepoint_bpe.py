from time import time, process_time
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import calculate_tokenizer_metrics
from constants import UNK_TOKEN, VOCAB_SIZE, BPE_SPECIAL_TOKENS, SAMPLE_TEXT
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths

paths = Paths()

# Define the tokenizer
codepoint_bpe = Tokenizer(BPE(unk_token=UNK_TOKEN))

# Normalizer, as previously defined in the `TamBert Data Analysis Notebook` file
codepoint_bpe.normalizer = normalizers.Sequence([normalizers.NFC(),
                                                 normalizers.Replace(Regex(r",+"), ","),
                                                 normalizers.Replace(Regex(r"\.+"), "."),
                                                 normalizers.Replace(Regex(r"'+"), "'"),
                                                 normalizers.Replace(Regex(r"-+"), "-")])

# Pre-tokenizer — Metaspace: bakes a word-boundary marker (▁)
# directly into token text, so decoding survives arbitrary subword splitting
codepoint_bpe.pre_tokenizer = pre_tokenizers.Metaspace()

print("Pre-tokenization process:")
print(codepoint_bpe.pre_tokenizer.pre_tokenize_str(SAMPLE_TEXT))

# Trainer
codepoint_bpe_trainer = BpeTrainer(special_tokens=BPE_SPECIAL_TOKENS, vocab_size=VOCAB_SIZE)

# Train
start_cpu_time, start_wall_time = process_time(), time()
codepoint_bpe.train([str(paths.train)], trainer=codepoint_bpe_trainer)
end_cpu_time, end_wall_time = process_time(), time()
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Decoder — must match the pre-tokenizer's marker scheme
codepoint_bpe.decoder = decoders.Metaspace()

# Evaluate
train_metrics = calculate_tokenizer_metrics(codepoint_bpe, paths.test)

# Post-processor — BERT [CLS]/[SEP] structure, set after training since it
# needs real token IDs from the trained vocab
codepoint_bpe.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", codepoint_bpe.token_to_id("[CLS]")),
        ("[SEP]", codepoint_bpe.token_to_id("[SEP]")),
    ],
)

# Report
print(f"Training completed in {cpu_time_taken:.2f}s (CPU) / {wall_time_taken:.2f}s (wall).")
print(f"Vocabulary size: {len(codepoint_bpe.get_vocab())}")
print(f"Fertility on test data: {train_metrics['fertility']:.4f}")
print(f"OOV rate on test data: {train_metrics['oov_rate']:.4f}")

encoded = codepoint_bpe.encode(SAMPLE_TEXT)
print(f"Encoding (with [CLS]/[SEP]): {encoded.tokens}")
print(f"Decoded: {codepoint_bpe.decode(encoded.ids)!r}")

# Save
out_dir = paths.tokenizer_name_generator('01_metaspace_codepoint_bpe')
codepoint_bpe.save(str(out_dir / 'tokenizer.json'))


with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fertility': train_metrics['fertility'],
        'oov_rate': train_metrics['oov_rate'],
        'vocab_size': len(codepoint_bpe.get_vocab()),
        'wall_time_seconds': wall_time_taken,
        'cpu_time_seconds': cpu_time_taken,
        'sample_text': {
            'raw': SAMPLE_TEXT,
            'encoded_tokens': encoded.tokens,
            'decoded': codepoint_bpe.decode(encoded.ids)
        }
    }, f, indent=2, ensure_ascii=False)

print(f"Saved to {out_dir}")