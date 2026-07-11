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

      # Invalid lines
      token_count = len(line.split())
      if token_count < min_tokens_per_line:
        invalid_lines += 1
        invalid_file.write(line+NEW_LINE)

      if not line:
        continue

      # Contamination calculations
      total_chars += len(line)
      disallowed = DISALLOWED.findall(line)
      if disallowed:
        disallowed_chars += len(disallowed)
        cont_file.write(line+NEW_LINE)

      # Error Anomaly calculation
      anomaly, error = is_anomalous_line(line)
      if anomaly:
        anomalous_lines += 1
        en_file.write(error+','+line+NEW_LINE)
      
      # Duplicate Ratio
      if line in sbf:
        duplicates += 1
        dup_file.write(line+NEW_LINE)
      else:
        sbf.add(line)

      # Non-Alphabetic character density
      non_alpha_chars += sum(1 for _ in NON_ALPHA.findall(line)) 

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

  return {'contamination_rate': contamination_rate,
          'non_alpha_rate': non_alpha_rate,
          'encoding_anomaly_rate': encoding_anomaly_rate,
          'duplicate_ratio': duplicate_ratio,
          'invalid_line_rate': invalid_line_rate}
  
def main():
  paths = Paths()
  hygiene_metrics = [ 'h1_contamination_', 'h2_encoding_anomaly_', 'h3_duplicate_ratio_', 'h5_invalid_lines_' ]
  results = []
  
  # Calculate corpus hygiene metrics for individual files
  for path in paths.metrics_targets():
    print(f"Calculating hygiene metrics for {path.name}...")
    metrics = calculate_hygine_metrcs(path, metrics_data=paths.metrics, hygiene_metrics=hygiene_metrics)
    metrics['source'] = path.name
    results.append(metrics)
    print(metrics)
    print()
    
  # Save the results to a CSV file
  summary_df = pd.DataFrame(results)  
  summary_df.to_csv(paths.hygiene_summary, index=False)

if __name__ == "__main__":
  main()