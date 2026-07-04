from pathlib import Path
import re
import unicodedata
from pybloom_live import ScalableBloomFilter
import regex
from utils import DISALLOWED, NON_ALPHA, NEW_LINE, REPLACEMENT_CHAR, has_orphaned_combining_mark, is_anomalous_line

def setup_environment():
    """
    Sets up the environment by creating necessary directories for storing hygiene metrics.
    """
    # Setup the Paths for data storage
    data = Path('data')
    data.mkdir(exist_ok=True, parents=True)
    print('Data folder created...')
    metrics_data = data / Path('metrics')
    metrics_data.mkdir(exist_ok=True, parents=True)
    print('Metrics folder created...')
    
    return metrics_data
# error storage files
contamination_txt = 'h1_contamination'
encoding_anomaly_txt = 'h2_encoding_anomaly'
duplicate_ratio_txt = 'h3_duplicate_ratio'
invalid_lines_txt = 'h5_invalid_lines'

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

def calculate_hygine_metrcs(file_path, error_rate=0.001, min_tokens_per_line=2, initial_capacity=1000, metrics_data=None):

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
      open(f'{metrics_data / contamination_txt}_{file_path.split("/")[-1]}', 'w', encoding='utf-8') as cont_file,
      open(f'{metrics_data / encoding_anomaly_txt}_{file_path.split("/")[-1]}', 'w', encoding='utf-8') as en_file,
      open(f'{metrics_data / duplicate_ratio_txt}_{file_path.split("/")[-1]}', 'w', encoding='utf-8') as dup_file,
      open(f'{metrics_data / invalid_lines_txt}_{file_path.split("/")[-1]}', 'w', encoding='utf-8') as invalid_file,
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
    contamination_rate = disallowed_chars / total_chars
    non_alpha_rate = non_alpha_chars / total_chars
  else:
    contamination_rate = non_alpha_rate = 0.0

  if total_lines:
    encoding_anomaly_rate = (anomalous_lines / total_lines) * 100
    duplicate_ratio = duplicates / total_lines 
    invalid_line_rate = (invalid_lines / total_lines) * 100
  else:
    encoding_anomaly_rate = duplicate_ratio = invalid_line_rate = 0

  return {'contamination_rate': contamination_rate,
          'non_alpha_rate': non_alpha_rate,
          'encoding_anomaly_rate': encoding_anomaly_rate,
          'duplicate_ratio': duplicate_ratio,
          'invalid_line_rate': invalid_line_rate}
  
def main():
  train_txt = 'data/corpus/train.txt'
  test_txt = 'data/corpus/test.txt'
  
  project_madurai_txt = 'data/cleaned/project_madurai_extracted.txt'
  tamil_wiki_txt = 'data/cleaned/tamil_wiki_extracted.txt'
  tamil_cc100_txt = 'data/cleaned/tamil_cc100_extracted.txt'
  
  # Calculate corpus hygiene metrics for individual files
  for path in [project_madurai_txt, tamil_wiki_txt, tamil_cc100_txt, train_txt, test_txt]:
    metrics = calculate_hygine_metrcs(path, metrics_data=setup_environment())
    print(f"Hygiene metrics for {path}:")
    print(metrics)
    print()
    