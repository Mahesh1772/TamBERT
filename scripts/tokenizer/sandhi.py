import regex as re
from dataclasses import dataclass
from typing import List, Tuple
from typing import Set 

@dataclass
class Rule:
    pattern: re.Pattern

from ta_sandhi_rules import TA_RULES_CORE
TA_PHONOLOGICAL_RULES: List[Rule] = TA_RULES_CORE

def _rule_boundary(m) -> int:
    if m.lastindex and m.lastindex >= 2:
        return m.start(2)
    return m.end()

def _phonological_boundaries(text: str, rules: List[Rule]) -> Set[int]:
    bounds: Set[int] = set()
    for r in rules:
        for m in r.pattern.finditer(text):   # scan only, no .sub(), no rewriting
            bounds.add(_rule_boundary(m))
    return bounds

_WS_SPLIT_RE = re.compile(r"\S+|\s+")

def _whitespace_boundaries(text: str) -> Set[int]:
    # Every \S<->\s transition. Correct, non-destructive replacement for rule J.
    return {m.start() for m in _WS_SPLIT_RE.finditer(text)}

def sandhi_split(text: str, lang: str = "ta") -> List[Tuple[str, Tuple[int, int]]]:
    """
    Same signature and return shape as before: [(token, (start, end)), ...].
    Guarantees text[start:end] == token for every tuple — the old version did not.
    """
    lang = lang.lower()
    if lang in ("en", "english"):
        rules: List[Rule] = []
    else:
        # "ta"/"tamil"/"mix"/"cmix" all use the same rules — every TA_RULES
        # pattern only matches Tamil-range characters, so scanning a mixed
        # string directly is safe; no chunk pre-splitting needed.
        rules = TA_PHONOLOGICAL_RULES

    bounds = _whitespace_boundaries(text)
    if rules:
        bounds |= _phonological_boundaries(text, rules)
    bounds.add(0)
    bounds.add(len(text))

    cut_points = sorted(bounds)
    tokens: List[Tuple[str, Tuple[int, int]]] = []
    for start, end in zip(cut_points, cut_points[1:]):
        if end > start:
            tokens.append((text[start:end], (start, end)))
    return tokens

def sandhi_mark_boundaries(text: str, lang: str = "ta", marker: str = "⟂") -> str:
    """
    Insert `marker` ONLY at genuine phonological (sandhi) boundaries —
    i.e. the same cut points sandhi_split derives from TA_PHONOLOGICAL_RULES.
    Plain whitespace-only word boundaries are left completely untouched
    (no marker inserted, whitespace itself unmodified), unlike the old
    "⟂".join(chunks) approach which stamped a marker at every single cut
    point sandhi_split produces, whitespace-only ones included.

    Property: text with all `marker` occurrences removed == the original
    `text`, exactly (marker is pure insertion, nothing else is touched).
    """
    lang = lang.lower()
    if lang in ("en", "english"):
        return text

    bounds = {b for b in _phonological_boundaries(text, TA_PHONOLOGICAL_RULES)
              if 0 < b < len(text)}  # drop no-op edges (start/end of string)
    if not bounds:
        return text

    pieces: List[str] = []
    prev = 0
    for b in sorted(bounds):
        pieces.append(text[prev:b])
        pieces.append(marker)
        prev = b
    pieces.append(text[prev:])
    return "".join(pieces)