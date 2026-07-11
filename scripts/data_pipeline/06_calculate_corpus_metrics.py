from pathlib import Path
import sys, unicodedata
import numpy as np
from collections import defaultdict
from pybloom_live import ScalableBloomFilter
from array import array
from scipy.stats import entropy as scipy_entropy
from utils import DISALLOWED, NON_ALPHA, NEW_LINE, REPLACEMENT_CHAR
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from paths import Paths

def is_combining_mark(ch: str) -> bool:
    return unicodedata.category(ch) in ('Mn', 'Mc', 'Me')

def has_orphaned_combining_mark(line: str) -> bool:
    """A combining mark is orphaned if nothing precedes it that can
    serve as a base (i.e. it's at the start of the line, or the
    previous char is whitespace/control, or was itself unresolved)."""
    prev_is_base = False
    for ch in line:
        if is_combining_mark(ch):
            if not prev_is_base:
                return True
            # stacked marks (e.g. base + 2 diacritics) stay valid;
            # don't reset prev_is_base here
        elif ch.isspace() or unicodedata.category(ch) in ('Cc', 'Cf'):
            prev_is_base = False
        else:
            prev_is_base = True
    return False

def is_anomalous_line(line: str):
    """True anomalies only. NFC mismatch alone is NOT flagged here —
    that's normal Tamil composition variance, not corruption. It should
    be fixed by normalizing, not by discarding the line."""
    if REPLACEMENT_CHAR in line:
        return True, 'replacement char'
    if has_orphaned_combining_mark(line):
        return True, 'orphaned mark'
    return False, None

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

def calculate_corpus_metrics(file_path, error_rate=0.001, min_tokens_per_line=2, initial_capacity=1000000, metrics_data=None, hygiene_metrics=None):

  sbf = ScalableBloomFilter(initial_capacity=initial_capacity, error_rate=error_rate)

  total_chars = 0
  disallowed_chars = 0
  non_alpha_chars = 0
  total_lines = 0
  invalid_lines = 0
  duplicates = 0
  anomalous_lines = 0
  sent_len = defaultdict(int)
  total_words = 0
  vocab = {}
  ids = array('I')

  with(
      open(file_path, 'r', encoding='utf-8') as f,
      open(f'{metrics_data / hygiene_metrics[0]}{file_path.name}', 'w', encoding='utf-8') as cont_file,
      open(f'{metrics_data / hygiene_metrics[1]}{file_path.name}', 'w', encoding='utf-8') as en_file,
      open(f'{metrics_data / hygiene_metrics[2]}{file_path.name}', 'w', encoding='utf-8') as dup_file,
      open(f'{metrics_data / hygiene_metrics[3]}{file_path.name}', 'w', encoding='utf-8') as invalid_file,
  ):
    for line in f:
      total_lines += 1
      line = line.rstrip('\n')

      # H5.Invalid lines
      token_count = len(line.split())
      if token_count < min_tokens_per_line:
        invalid_lines += 1
        invalid_file.write(line+NEW_LINE)

      if not line:
        continue

      # H1.Contamination calculations
      total_chars += len(line)
      disallowed = DISALLOWED.findall(line)
      if disallowed:
        disallowed_chars += len(disallowed)
        cont_file.write(line+NEW_LINE)

      # H2.Encoding Anomaly calculation
      anomaly, error = is_anomalous_line(line)
      if anomaly:
        anomalous_lines += 1
        en_file.write(error+','+line+NEW_LINE)
      
      # H3.Duplicate Ratio
      if line in sbf:
        duplicates += 1
        dup_file.write(line+NEW_LINE)
      else:
        sbf.add(line)

      # H4.Non-Alphabetic character density
      non_alpha_chars += sum(1 for _ in NON_ALPHA.findall(line))
      
      ## Corpus Profiling Metrics
      
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
      unigram_entropy = scipy_entropy(freq, base=2) if len(freq) > 1 else 0.0

  if total_chars:
    contamination_rate = (disallowed_chars / total_chars) * 100
    non_alpha_rate = (non_alpha_chars / total_chars) * 100
  else:
    contamination_rate = non_alpha_rate = 0.0

  if total_lines:
    encoding_anomaly_rate = (anomalous_lines / total_lines) * 100
    duplicate_ratio = (duplicates / total_lines) * 100
    invalid_line_rate = (invalid_lines / total_lines) * 100
  else:
    encoding_anomaly_rate = duplicate_ratio = invalid_line_rate = 0

  hygiene = {'contamination_rate': contamination_rate,
          'non_alpha_rate': non_alpha_rate,
          'encoding_anomaly_rate': encoding_anomaly_rate,
          'duplicate_ratio': duplicate_ratio,
          'invalid_line_rate': invalid_line_rate}
  
  corpus_metrics = {'total_words': total_words,
          'sentence_length_distribution': dict(sent_len),
          'ttr': ttr,
          'mtld': mtld,
          'unigram_entropy': unigram_entropy}
    
  return {**hygiene, **corpus_metrics}
  
def main():
  paths = Paths()
  hygiene_metrics = [ 'h1_contamination_', 'h2_encoding_anomaly_', 'h3_duplicate_ratio_', 'h5_invalid_lines_' ]
  hygiene_results = []
  corpus_metrics_results = []
  sent_len_dist_results = {}
  
  # Calculate corpus hygiene metrics for individual files
  for path in paths.metrics_targets():
    print(f"Calculating hygiene metrics for {path.name}...")
    hygiene, corpus = calculate_corpus_metrics(path, metrics_data=paths.metrics, hygiene_metrics=hygiene_metrics)
    hygiene['source'] = path.name
    corpus['source'] = path.name
    hygiene_results.append(hygiene)
    corpus_metrics_results.append(corpus)
    print(hygiene)
    print(corpus)
    
    sent_len_dist = corpus.pop('sentence_length_distribution')
    sent_len_dist_results[path.name] = sent_len_dist
    png_path = save_sentence_length_distribution(sent_len_dist, path.stem, paths.metrics)
    print(f"Saved sentence length distribution plot to {png_path}")
    
    print()
    
  # Save the results to a CSV file
  
  sent_len_summary = []
  for source, dist in sent_len_dist_results.items():
    for length, count in dist:
      sent_len_summary.append({'source': source, 'sentence_length': length, 'count': count})
  
  summary_df = pd.DataFrame(hygiene_results)  
  summary_df.to_csv(paths.hygiene_summary, index=False)

  sent_len_df = pd.DataFrame(sent_len_summary)
  sent_len_df.to_csv(paths.sentence_length_summary, index=False)

  corpus_df = pd.DataFrame(corpus_metrics_results)
  corpus_df.to_csv(paths.corpus_summary, index=False)
  

if __name__ == "__main__":
  main()