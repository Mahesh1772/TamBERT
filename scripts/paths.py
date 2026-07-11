from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Paths:
    """
    A class to manage and provide paths for various directories and files used in the project.
    """
    root: Path = PROJECT_ROOT
    data : Path = field(init=False)
    raw : Path = field(init=False)
    cleaned : Path = field(init=False)
    metrics : Path = field(init=False)
    corpus : Path = field(init=False)
    
    project_madurai : Path = field(init=False)
    tamil_wiki : Path = field(init=False)
    tamil_wiki_long_lines : Path = field(init=False)
    tamil_cc100 : Path = field(init=False)
    merged_deduped : Path = field(init=False)
    train : Path = field(init=False)
    test : Path = field(init=False)

    def __post_init__(self):
        # Ensure that all directories exist
        self.data = self.root / 'data'
        self.raw = self.data / 'raw'
        self.cleaned = self.data / 'cleaned'
        self.metrics = self.data / 'metrics'
        self.corpus = self.data / 'corpus'
        
        for directory in [self.data, self.raw, self.cleaned, self.metrics, self.corpus]:
            directory.mkdir(exist_ok=True, parents=True)
            print(f"Directory created or already exists: {directory}")
            
        # Define file paths
        self.project_madurai = self.cleaned / 'project_madurai_extracted.txt'
        self.project_madurai = self.cleaned / 'project_madurai_extracted_above_threshold.txt'
        
        self.wiki_dump = self.raw / 'tamilwiki-latest-pages-articles.xml.bz2'
        self.tamil_wiki = self.cleaned / 'tamil_wiki_extracted.txt'
        self.tamil_wiki_long_lines = self.cleaned / 'tamil_wiki_extracted_long_lines.txt'
        
        self.cc100_file = self.raw / 'tamil_cc100.txt.xz'
        self.tamil_cc100 = self.cleaned / 'tamil_cc100_extracted.txt'
        
        self.merged = self.cleaned / 'merged.txt'
        self.dropped_duplicates = self.cleaned / 'dropped_duplicates.txt'
        self.merged_deduped = self.cleaned / 'merged_deduped.txt'
        self.train = self.corpus / 'train.txt'
        self.test = self.corpus / 'test.txt'
    
    def metrics_targets(self):
        """
        Returns a list of file paths that are targets for hygiene metrics.
        """
        return [
            self.project_madurai,
            self.tamil_wiki,
            self.tamil_cc100,
            self.test,
            self.train,
            self.merged_deduped,
            self.tamil_wiki_long_lines
        ]
        