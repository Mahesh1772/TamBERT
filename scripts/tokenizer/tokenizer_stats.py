import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import Paths
from pipeline import load_config
from metrics import calculate_tokenizer_metrics

# each family's config.yaml, relative to this file
FAMILY_CONFIGS = ["bpe/config.yaml", "unigram/config.yaml", "wordpiece/config.yaml"]


# 1. collect every tokenizer variant's metrics.json + tokenizer.json
def discover_tokenizer_dirs(paths):
    print(f"Discovering tokenizer variants in {paths.tokenizer}...")
    variants = {}
    for sub in Path(paths.tokenizer).iterdir():
        if sub.is_dir():
            variants[sub.name] = {
                "metrics": sub / "metrics.json",
                "tokenizer": sub / "tokenizer.json",
            }
    print(f"Found {len(variants)} tokenizer variants.")
    return variants


# 2. build the fertility / oov / vocab_size table
def build_metrics_dataframe(variants):
    print("Building metrics dataframe...")
    rows = []
    for name, files in variants.items():
        with open(files["metrics"], encoding="utf-8") as f:
            data = json.load(f)
        data["file"] = name
        rows.append(data)
    df = pd.DataFrame(rows).set_index("file")
    print(df[["fertility", "oov_rate", "vocab_size"]])
    return df


# 3. one bar chart per metric, values labelled on top
def plot_metric_bars(df, out_dir):
    print("Plotting metric bars...")
    plot_df = df[["fertility", "oov_rate", "vocab_size"]].sort_values("fertility")
    formats = {"fertility": "%.3f", "oov_rate": "%.6f", "vocab_size": "%.0f"}
    for col in plot_df.columns:
        ax = plot_df[col].plot(kind="bar", figsize=(12, 6), legend=False, title=col)
        ax.bar_label(ax.containers[0], fmt=formats[col])
        plt.tight_layout()
        plt.savefig(out_dir / f"tokenizer_{col}.png")
        plt.close()


# variant name -> correct test file for the SAVED tokenizer.json
# (relabelled grapheme variants expect raw text, not placeholder text)
def build_test_file_map(paths):
    base = Path(__file__).resolve().parent
    test_map = {}
    for rel_path in FAMILY_CONFIGS:
        for cfg in load_config(base / rel_path):
            use_grapheme = cfg.get("use_grapheme", False)
            use_sandhi = cfg.get("use_sandhi", False)
            relabelled = cfg.get("relabel_vocab_after_save", False)
            if use_grapheme and relabelled:
                attr = "test_sandhi_marked" if use_sandhi else "test"
            else:
                attr = cfg["test_attr"]
            test_map[cfg["name"]] = getattr(paths, attr)
    return test_map


# 4. token length distribution per variant, zoomed and full-range as separate pngs
def plot_length_distributions(variants, test_file_map, out_dir):
    print("Plotting length distributions...")
    for name, files in variants.items():
        test_file = test_file_map.get(name)
        if test_file is None:
            continue
        tok = Tokenizer.from_file(str(files["tokenizer"]))
        
        print(f"Calculating token lengths for {name} using {test_file}...")
        
        lengths = np.array([
            len(tok.encode(line, add_special_tokens=False).ids)
            for line in open(test_file, encoding="utf-8")
        ])

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(lengths, bins=50, range=(0, 200))
        ax.set(title=f"{name} - zoomed (0-200)", xlabel="Tokens per line", ylabel="Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / f"token_length_{name}_zoomed.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(lengths, bins=50)
        ax.set_yscale("log")
        ax.set(title=f"{name} - full range, log y", xlabel="Tokens per line", ylabel="Frequency (log)")
        plt.tight_layout()
        plt.savefig(out_dir / f"token_length_{name}_full_log.png")
        plt.close()

        print(f"Token length stats for {name}:")
        print(pd.Series(lengths).describe(percentiles=[.5, .75, .9, .95, .999]))


# isolated vs fused usage of the ⟂ marker for one tokenizer
def sandhi_marker_usage(tokenizer, test_file):
    vocab = tokenizer.get_vocab()
    marker_ids = {id_: tok for tok, id_ in vocab.items() if "⟂" in tok}
    isolated_ids = {id_ for id_, tok in marker_ids.items() if tok.lstrip("▁") == "⟂"}

    usage = Counter()
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for id_ in tokenizer.encode(line, add_special_tokens=False).ids:
                if id_ in marker_ids:
                    usage[id_] += 1

    isolated = sum(c for i, c in usage.items() if i in isolated_ids)
    fused = sum(c for i, c in usage.items() if i not in isolated_ids)
    return isolated, fused


def plot_marker_usage(usage_df, out_path, title):
    ax = usage_df.plot(kind="bar", figsize=(10, 6), title=title)
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# 5. marker usage across sandhi variants, split codepoint / grapheme / combined
def run_sandhi_analysis(variants, test_file_map, out_dir):
    codepoint_rows, grapheme_rows = {}, {}
    for name, files in variants.items():
        if "sandhi" not in name:
            continue
        test_file = test_file_map.get(name)
        if test_file is None:
            continue
        tok = Tokenizer.from_file(str(files["tokenizer"]))
        isolated, fused = sandhi_marker_usage(tok, test_file)
        target = grapheme_rows if "grapheme" in name else codepoint_rows
        target[name] = {"isolated": isolated, "fused": fused}
        print(f"{name}: isolated={isolated}, fused={fused}")

    codepoint_df = pd.DataFrame(codepoint_rows).T
    grapheme_df = pd.DataFrame(grapheme_rows).T
    combined_df = pd.concat([codepoint_df, grapheme_df])

    if not codepoint_df.empty:
        plot_marker_usage(codepoint_df, out_dir / "sandhi_marker_usage_codepoint.png",
                           "Sandhi marker usage - codepoint")
    if not grapheme_df.empty:
        plot_marker_usage(grapheme_df, out_dir / "sandhi_marker_usage_grapheme.png",
                           "Sandhi marker usage - grapheme")
    if not combined_df.empty:
        plot_marker_usage(combined_df, out_dir / "sandhi_marker_usage_combined.png",
                           "Sandhi marker usage - all sandhi variants")


# 6. mBERT comparison
def compare_with_mbert(paths, out_dir):
    mbert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
    metrics = calculate_tokenizer_metrics(mbert.backend_tokenizer, paths.test, unknown_token="[UNK]")
    with open(out_dir / "mbert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"mBERT: fertility={metrics['fertility']:.4f} oov_rate={metrics['oov_rate']:.4f}")
    return metrics


def main():
    paths = Paths()
    out_dir = paths.tokenizer_name_generator("tokenizer_statistics")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = discover_tokenizer_dirs(paths)
    df = build_metrics_dataframe(variants)
    plot_metric_bars(df, out_dir)

    test_file_map = build_test_file_map(paths)
    plot_length_distributions(variants, test_file_map, out_dir)
    run_sandhi_analysis(variants, test_file_map, out_dir)
    compare_with_mbert(paths, out_dir)


if __name__ == "__main__":
    main()