from pathlib import Path
import numpy as np
from collections import defaultdict
from array import array
from scipy.stats import entropy as scipy_entropy
from utils import create_directories, NON_ALPHA, REPLACEMENT_CHAR, DISALLOWED, NEW_LINE

def setup_environment():
    """
    Sets up the environment by creating necessary directories for storing corpus profiling metrics.
    """
    # Setup the Paths for data storage
    
    data, _, cleaned_data = create_directories()  # Ensure the data directories exist
    metrics_data = data / Path('metrics')
    metrics_data.mkdir(exist_ok=True, parents=True)
    print('Metrics folder created...')
    
    train_txt = data / Path('corpus/train.txt')
    test_txt = data / Path('corpus/test.txt')
    
    project_madurai_txt = cleaned_data / Path('project_madurai_extracted.txt')
    tamil_wiki_txt = cleaned_data / Path('tamil_wiki_extracted.txt')
    tamil_cc100_txt = cleaned_data / Path('tamil_cc100_extracted.txt')
    
    file_paths = [project_madurai_txt, tamil_wiki_txt, tamil_cc100_txt, test_txt, train_txt]

    hygiene_metrics = [ 'h1_contamination_', 'h2_encoding_anomaly_', 'h3_duplicate_ratio_', 'h5_invalid_lines_' ]
       
    return metrics_data, file_paths, hygiene_metrics

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
