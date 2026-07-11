from time import time, process_time
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from metrics import BPE_SPECIAL_TOKENS, SAMPLE_TEXT, UNK_TOKEN, VOCAB_SIZE, calculate_tokenizer_metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/ — 3 levels up
from paths import Paths

print(str(Path(__file__).resolve().parents[2]))

paths = Paths()

# Define the tokenizer
codepoint_bpe = Tokenizer(BPE(unk_token=UNK_TOKEN))

# Define the normalizer
codepoint_bpe.normalizer = normalizers.NFC()

# Define the pre-tokenizer
codepoint_bpe.pre_tokenizer = pre_tokenizers.Whitespace() 

# Visualize the pre-tokenization process
print("Pre-tokenization process:")
print(codepoint_bpe.pre_tokenizer.pre_tokenize_str(SAMPLE_TEXT))

# Define the trainer
codepoint_bpe_trainer = BpeTrainer(special_tokens=BPE_SPECIAL_TOKENS, vocab_size=VOCAB_SIZE)

# Train the tokenizer on the train data
start_cpu_time, start_wall_time = time(), process_time()
codepoint_bpe.train([str(paths.train)], trainer=codepoint_bpe_trainer)
end_cpu_time, end_wall_time = time(), process_time()

# Calculate the time taken for training
cpu_time_taken = end_cpu_time - start_cpu_time
wall_time_taken = end_wall_time - start_wall_time

# Evaluate the tokenizer on the test data
train_metrics = calculate_tokenizer_metrics(codepoint_bpe, paths.test)  

# Print statistics about the training process
print(f"Training completed in {cpu_time_taken:.2f} seconds (CPU time) and {wall_time_taken:.2f} seconds (wall time).")
print(f"Tokenizer vocabulary size: {len(codepoint_bpe.get_vocab())}")
print(f"Tokenizer fertility on test data: {train_metrics['fertility']:.4f}")
print(f"Tokenizer OOV rate on test data: {train_metrics['oov_rate']:.4f}")
print(f"Tokenizer encoding: {codepoint_bpe.encode(SAMPLE_TEXT).tokens}")