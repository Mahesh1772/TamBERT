import grapheme
from tokenizers import NormalizedString, PreTokenizedString
import regex as re

_GRAPHEME_RE = re.compile(r"\X")   # native C-level extended-grapheme-cluster matching

class GraphemePreTokenizer(PreTokenizedString):
    def pre_tokenize(self, pretok: PreTokenizedString):
        
        def split_on_graphemes(normalized_string: NormalizedString):
            text = str(normalized_string)
            return [normalized_string[m.start():m.end()] for m in _GRAPHEME_RE.finditer(text)]

        pretok.split(split_on_graphemes)