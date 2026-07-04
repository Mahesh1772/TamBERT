import shutil, bz2, os, random, fileinput
from pathlib import Path
from tqdm import tqdm
from utils import create_directories, reservoir_sample, TAMIL_CHARACTERS

def setup_environment():
    """
    Sets up the environment by creating necessary directories and downloading the Tamil Wikipedia dump.
    """
    # Setup the Paths for data storage
    data, _, cleaned_data = create_directories()

    # Setup paths for Merged text file and the train/test files
    merged_text_file = cleaned_data / 'merged.txt'
    
    if merged_text_file.exists():
        print(f"Merged text file already exists at {merged_text_file}. It will be overwritten.")
        merged_text_file.unlink()  # Remove the existing file to avoid appending to it
    
    # Setup Corpus directory for storing the train/test files
    corpus_dir = data / 'corpus'
    corpus_dir.mkdir(exist_ok=True, parents=True)
    
    train_file = corpus_dir / 'train.txt'
    test_file = corpus_dir / 'test.txt'

    return merged_text_file, train_file, test_file, cleaned_data


def merge_text_files(input_dir:Path, output_file:Path):
    """
    Merges all text files in the input directory into a single output file.

    Args:
        input_dir (Path): The directory containing text files to merge.
        output_file (Path): The path to the output file where merged content will be saved.
    """
    
    cleaned_text_files = list(input_dir.glob('*.txt'))
    
    # Calc size for progress bar
    total_size = sum(os.path.getsize(f) for f in cleaned_text_files)
    
    # Calc lines skipped based on quality check (presense of at least one Tamil character)
    lines_skipped = lines_written = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Merging text files") as pbar:
            input_lines = fileinput.input(cleaned_text_files, encoding='utf-8')
            
            # Full for loop for progress bar update
            for line in input_lines:
                if TAMIL_CHARACTERS.search(line):
                    outfile.write(line)
                    lines_written += 1
                else:
                    lines_skipped += 1
                pbar.update(len(line.encode('utf-8')))
    
    print(f"Lines written: {lines_written}, Lines skipped: {lines_skipped}")
    print(f"Merged text files into {output_file} with a total size of {os.path.getsize(output_file)/1024/1024/1024:.2f} GB.")
    

def create_file_assignment_list(input_file:Path, test_split=0.1):
    """
    Creates a list of assignments ('train' or 'test') for each line in the input file based on the specified test split.

    Args:
        input_file (Path): The path to the input text file.
        test_split (float): The fraction of lines to assign to the test set (default is 0.1).
    
    Returns:
        list: A list of assignments corresponding to each line in the input file.
    """
    
    # Read the total number of lines in the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Calculate the number of lines for the test set
    test_size = int(total_lines * test_split)
    train_size = total_lines - test_size
    
    # Create a list of assignments
    line_assignments = ['train'] * train_size + ['test'] * test_size
    
    # Shuffle the assignments to randomize train/test distribution
    random.seed(42)  # For reproducibility
    random.shuffle(line_assignments)  # Shuffle with a fixed seed for reproducibility
    
    return line_assignments, total_lines, train_size, test_size


def create_train_test_files(merged_file:Path, train_file:Path, test_file:Path, test_split=0.1):
    """
    Creates train and test files from the merged text file using reservoir sampling.

    Args:
        merged_file (Path): The path to the merged text file.
        train_file (Path): The path to the output train file.
        test_file (Path): The path to the output test file.
        test_split (float): The fraction of lines to sample for the test file (default is 0.1).
    """
    
    print(f"Creating train and test files from {merged_file} with a test split of {test_split*100:.1f}%...")
    # Obtain the total number of lines in the merged file
    line_indices, total_lines, train_size, test_size = create_file_assignment_list(merged_file, test_split=test_split)
    
    print(f"Total lines in merged file: {total_lines}, Train size: {train_size}, Test size: {test_size}")
    # Read form the merged file and write to train and test files based on the shuffled indices
    with(
        open(merged_file, 'r', encoding='utf-8') as infile,
        open(train_file, 'w', encoding='utf-8') as train_outfile,
        open(test_file, 'w', encoding='utf-8') as test_outfile
    ):
        with tqdm(total=total_lines, unit='lines', desc="Creating train/test files") as pbar:
            for line, assignment in zip(infile, line_indices):
                if assignment == 'train':
                    train_outfile.write(line)
                else:
                    test_outfile.write(line)
                pbar.update(1)
    
    print(f'Created train file:{train_file} with {train_size} lines.')
    print(f'Created test file:{test_file} with {test_size} lines.')


def main():
    # Setup environment and get paths for merged, train, and test files
    merged_file, train_file, test_file, cleaned_data = setup_environment()
    
    # Merge all cleaned text files into a single merged file
    merge_text_files(cleaned_data, merged_file)
    
    # Create train and test files from the merged file
    create_train_test_files(merged_file, train_file, test_file, test_split=0.1)
    
    # Print sample lines from the train and test files for verification
    sampled_train_lines = reservoir_sample(train_file, k=5)
    sampled_test_lines = reservoir_sample(test_file, k=5)
    
    print("\nSampled lines from the train file:\n")
    for line in sampled_train_lines:
        print(line)
        
    print("\nSampled lines from the test file:\n")
    for line in sampled_test_lines:
        print(line)
        
main()