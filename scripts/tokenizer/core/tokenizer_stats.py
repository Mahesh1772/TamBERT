import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from paths import Paths
from tokenizer.core.pipeline import load_config
from tokenizer.core.metrics import calculate_tokenizer_metrics
from tokenizer.core.preprocess import corpus_attr, family_config_paths, saved_tokenizer_markings


# 1. collect every tokenizer variant's metrics.json + tokenizer.json
def discover_tokenizer_dirs(paths):
    variants = {}
    for sub in Path(paths.tokenizer).iterdir():
        metrics_file = sub / "metrics.json"
        tokenizer_file = sub / "tokenizer.json"
        if sub.is_dir() and metrics_file.exists() and tokenizer_file.exists():
            variants[sub.name] = {"metrics": metrics_file, "tokenizer": tokenizer_file}
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
    print("Plotting fertility / oov_rate / vocab_size bar charts...")
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
    # Config paths come from family_config_paths() rather than being built off __file__ here: this module lives
    # in scripts/tokenizer/core/ but the configs are one level up, and the local version resolved them against
    # core/ and so never found any of them.
    print("Building variant to test file map from family configs...")
    test_map = {}
    for config_path in family_config_paths():
        for cfg in load_config(config_path):
            # saved_tokenizer_markings folds in relabel_vocab_after_save, which is the whole subtlety here: a
            # relabelled grapheme variant's vocab is real Tamil again, so it has to be measured against the
            # NON-substituted file, not the *_grapheme_marked one its train_attr/test_attr names.
            use_sandhi, use_grapheme = saved_tokenizer_markings(cfg)
            test_map[cfg["name"]] = getattr(paths, corpus_attr(use_sandhi, use_grapheme, split="test"))
    return test_map


# 4. token length distribution per variant, zoomed and full-range as separate pngs
def plot_length_distributions(variants, test_file_map, out_dir):
    print("Computing token length distributions...")
    stats_rows = {}
    for name, files in variants.items():
        test_file = test_file_map.get(name)
        if test_file is None:
            continue
        print(f"  {name}")
        tok = Tokenizer.from_file(str(files["tokenizer"]))
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

        stats_rows[name] = pd.Series(lengths).describe(percentiles=[.5, .75, .9, .95, .999])

    stats_df = pd.DataFrame(stats_rows).T
    stats_df.to_csv(out_dir / "token_length_stats.csv")
    print(stats_df)
    return stats_df


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
    print("Running sandhi marker usage analysis...")
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
    combined_df.to_csv(out_dir / "sandhi_marker_usage.csv")

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
    print("Loading mBERT tokenizer for comparison...")
    mbert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
    metrics = calculate_tokenizer_metrics(mbert.backend_tokenizer, paths.test, unknown_token="[UNK]")
    with open(out_dir / "mbert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"mBERT: fertility={metrics['fertility']:.4f} oov_rate={metrics['oov_rate']:.4f}")
    return metrics


def main():
    paths = Paths()
    out_dir = paths.tokenizer_name_generator("tokenizer_statistics")

    variants = discover_tokenizer_dirs(paths)
    df = build_metrics_dataframe(variants)
    plot_metric_bars(df, out_dir)

    test_file_map = build_test_file_map(paths)
    plot_length_distributions(variants, test_file_map, out_dir)
    run_sandhi_analysis(variants, test_file_map, out_dir)
    compare_with_mbert(paths, out_dir)


if __name__ == "__main__":
    main()