from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
from array import array
from scipy.stats import entropy as scipy_entropy
from utils import create_directories, NEW_LINE
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from paths import Paths

def mtld_pass(ids, threshold=0.72):
  factors = 0
  types = set()
  count = 0

  for i in ids:
    count += 1
    types.add(i)

    # If ttr collabs below threshold
    if len(types)/count <= threshold:
      factors += 1
      types.clear()
      count = 0

  if count > 0:
    ttr = len(types)/count
    factors += (1-ttr) / (1-threshold)

  return len(ids) / factors if factors > 0 else 0.0

def calculate_corpus_metrics(file_name):

  sent_len = defaultdict(int)
  total_words = 0
  vocab = {}
  ids = array('I')

  with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.rstrip(NEW_LINE)
      line_parts = line.split()
      line_parts_len = len(line_parts)

      # 1. Total words/tokens in corpus
      total_words += line_parts_len

      # 2. Sentence length distribution
      sent_len[line_parts_len] += 1

      # 3/4. TTR and MLTD
      for tok in line_parts:
        idx = vocab.get(tok)
        if idx is None:
          idx = len(vocab)
          vocab[tok] = idx
        ids.append(idx)

  # 3. TTR calculation
  ttr = len(vocab)/ len(ids) if ids else 0.0

  # 4. MLTD calculation
  forward_pass = mtld_pass(ids, threshold=0.72)
  backward_pass = mtld_pass(ids[::-1], threshold=0.72)
  mtld = (forward_pass + backward_pass) / 2

  # 5. Unigram entorpy
  freq = np.bincount(ids)
  unigram_entropy = scipy_entropy(freq, base=2)

  return {'Word Count': total_words,
          'Sentence Length Distribution': sorted(sent_len.items()),
          'TTR': ttr,
          'MTLD': mtld,
          'Unigram Entropy': unigram_entropy}

def save_sentence_length_distribution(sent_len_dist, source_name, output_dir):
    """Save sentence-length distribution as a log-scale bar chart PNG.

    sent_len_dist: list of (length, count) tuples, e.g. sorted(sent_len.items())
    source_name: used for both the plot title and the output filename
    output_dir: Path to the metrics folder

    Returns the Path the PNG was saved to.
    """
    lengths, counts = zip(*sent_len_dist)
    plt.figure(figsize=(8, 4))
    plt.bar(lengths, counts, width=0.9)
    plt.yscale('log')
    plt.xlabel('Sentence length (tokens)')
    plt.ylabel('Count (log scale)')
    plt.title(f'Sentence Length Distribution — {source_name}')
    plt.tight_layout()
    png_path = output_dir / f'sent_len_dist_{source_name}.png'
    plt.savefig(png_path, dpi=120)
    plt.close()
    return png_path

def main():
    """
    Main function to calculate corpus profiling metrics for the specified text files.
    """
    paths = Paths()
    results = []
    sent_len_dist_results = {}

    # Calculate corpus profiling metrics for individual files
    for path in paths.metrics_targets():
        print(f"Calculating corpus metrics for {path.name}...")
        metrics = calculate_corpus_metrics(path)
        
        sent_len_dist = metrics.pop('Sentence Length Distribution')
        sent_len_dist_results[path.name] = sent_len_dist
        png_path = save_sentence_length_distribution(sent_len_dist, path.stem, paths.metrics)
        print(f"Saved sentence length distribution plot to {png_path}")
        
        metrics['source'] = path.name
        results.append(metrics)
        print(metrics)
        print()
    
    # Summarize and save results to CSV
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(paths.corpus_metrics_summary, index=False)
    
    sent_len_summary = []
    for source, dist in sent_len_dist_results.items():
        for length, count in dist:
            sent_len_summary.append({'source': source, 'sentence_length': length, 'count': count})

    sent_len_summary_df = pd.DataFrame(sent_len_summary)
    sent_len_summary_df.to_csv(paths.sentence_length_summary, index=False)

if __name__ == "__main__":
    main()