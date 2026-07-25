"""
Entry point for the Unigram family of the tokenizer ablation study.
Shared training logic lives in pipeline.py; this just points it at
this folder's config.yaml.

    python train_unigram_tokenizer.py
    python train_unigram_tokenizer.py --only 03_sandhi_codepoint_unigram
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import run_variants
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import Paths


def main():
    parser = argparse.ArgumentParser(description="Train Unigram tokenizer variants")
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'config.yaml'))
    parser.add_argument('--only', help="Train a single variant by name, e.g. 03_sandhi_codepoint_unigram")
    args = parser.parse_args()

    results = run_variants(args.config, Paths(), only=args.only)

    print("\n=== Summary ===")
    for name, m in results.items():
        print(f"{name}: fertility={m['fertility']:.4f}  oov_rate={m['oov_rate']:.4f}  vocab_size={m['vocab_size']}")


if __name__ == '__main__':
    main()
