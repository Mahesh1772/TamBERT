"""
Tamil சந்தி (sandhi) boundary rules — drop-in replacement for TA_RULES in sandhi.py.

SOURCES (consulted while building this list):
  - நன்னூல் எழுத்ததிகாரம், இயல் 3 "உயிரீற்றுப் புணரியல்" (நூற்பா 151-203, 53 sutras)
    https://ta.wikipedia.org/wiki/உயிரீற்றுப்_புணரியல்_(நன்னூல்)
    https://ta.wikisource.org/wiki/நன்னூல்_எழுத்ததிகாரம்_3._உயிரீற்றுப்புணரியல்
  - Tamil Virtual Academy, hg314 lesson (திசைப்பெயர்ப் புணர்ச்சி, மையீற்றுப் பண்புப்
    பெயர்ப் புணர்ச்சி, உடல்+உயிர், பூப்பெயர், தேங்காய், தனிக்குறில் முன் ஒற்று,
    உடம்படுமெய், வல்லினம் மிகும்/மிகா இடங்கள்)
    https://www.tamilvu.org/courses/hg300/hg314/html/hg314ple.htm
  - "நன்னூல் விதிகள்" numbered sutra-by-sutra series (blog), used to cross-check
    individual நூற்பாக்கள் (e.g. 162, 164, 186, 189-199)
    https://thamizhppanimanram.blogspot.com/

WHAT THIS FILE IS AND ISN'T
  நன்னூல் புணர்ச்சியியல் has ~107 sutras across the uyir-ending, mெய்-ending and
  urubu chapters. A large fraction of them are conditioned on WORD CLASS or
  GRAMMATICAL CATEGORY (imperative verbs, vocative case, numerals other than
  எட்டு/பத்து, uyartinai vs. அஃறிணை nouns, etc.) — a plain regex over characters
  cannot know a word's part of speech, so those sutras are NOT representable
  here and are intentionally left out rather than faked.
  What IS representable: (a) genuine surface phonological cues (vowel-vowel
  joins, glide insertion, consonant-vowel elision), and (b) a handful of
  CLOSED lexical sets (direction words, mை-adjectives, சுட்டு/அளவைச் சொற்கள்)
  that are finite enough to hard-code safely.

TWO TIERS, ON PURPOSE
  TA_RULES_CORE        — high precision. Vowel joins, glide insertion, the
                          general "consonant+vowel automatically joins" rule
                          (நூற்பா 204), case-suffix / verbal-participle joins,
                          and the small closed lexical sets. Rarely fires
                          inside a native, non-compound word.
  TA_RULES_AGGRESSIVE  — high recall, LOWER precision. Nasal+stop clusters
                          (ங்க/ஞ்ச/ண்ட/ந்த/ம்ப/ன்ற) and consonant gemination
                          (க்க/ச்ச/.../ற்ற/ன்ன) are real sandhi phenomena, but
                          they are ALSO just how Tamil natively spells those
                          clusters inside single, non-compound words (அன்பு,
                          தங்கம், இந்த, சிந்தனை, பள்ளி, கல்லூரி, நல்ல, வெள்ளை
                          all match one of these patterns internally with zero
                          sandhi happening). Include this tier only if your
                          downstream use tolerates that noise, or if you can
                          pair it with a lexicon/exception-word filter.

TA_RULES = TA_RULES_CORE + TA_RULES_AGGRESSIVE + [_WHITESPACE_RULE] preserves
the exact old drop-in shape sandhi.py expects (last element is the whitespace
rule, excluded via TA_RULES[:-1] to build TA_PHONOLOGICAL_RULES). If you want
the higher-precision behaviour, just do:
    TA_RULES = TA_RULES_CORE + [_WHITESPACE_RULE]
in sandhi.py instead.
"""
import regex as re
from dataclasses import dataclass
from typing import List


@dataclass
class Rule:
    pattern: re.Pattern


# Reusable character-class bodies (no stray spaces — the original file had a
# few "[... ஒஓஐ ஔ]"-style classes with accidental literal spaces baked in;
# those are bugs, cleaned up here).
_V = "அஆஇஈஉஊஎஏஐஒஓஔ"          # 12 independent vowels (உயிரெழுத்து)
_C = "கஙசஞடணதநபமயரலவழளறன"     # 18 base consonants (மெய்யெழுத்து, sans ஃ)
_MATRAS = "ாிீுூெேைொோௌ"       # 11 dependent vowel signs (உயிர்மெய் matras)

TA_RULES_CORE: List[Rule] = [

    # =====================================================================
    # A) உயிர் + உயிர் (vowel-vowel coalescence cues)
    # Two vowels landing next to each other across a boundary is itself
    # the trigger for a உடம்படுமெய் (glide) or கூட்டல் (merger) — mark the
    # join before the second vowel so nothing coalesces across it.
    # =====================================================================
    Rule(re.compile(r"(அ)(அ|ஆ)")),               # அ+அ/ஆ ā-type merger (பல+அரண்-style)
    Rule(re.compile(r"(அ)(இ|ஈ)")),               # அ+இ/ஈ e-like outcome
    Rule(re.compile(r"(அ)(உ|ஊ)")),               # அ+உ/ஊ o-like outcome
    Rule(re.compile(r"(அ)(எ|ஏ|ஒ|ஓ|ஐ|ஔ)")),       # அ+other vowel, generic join
    Rule(re.compile(rf"(இ|ஈ)([{_V}])")),          # இ/ஈ + any vowel: ய்-glide zone
    Rule(re.compile(rf"(உ|ஊ)([{_V}])")),          # உ/ஊ + any vowel: வ்-glide zone
    Rule(re.compile(rf"(எ|ஏ|ஒ|ஓ|ஐ|ஔ)([{_V}])")), # remaining vowel + vowel joins

    # =====================================================================
    # B) உடம்படுமெய் தோன்றல் (நூற்பா 162) — glide (ய்/வ்) insertion cues.
    # "இ ஈ ஐ வழி யவ்வும் ஏனை உயிர்வழி வவ்வும் ஏ முன் இவ்விருமையும்
    #  உயிர்வரின் உடம்படுமெய் யென்றாகும்." — மணி+அடித்தது → மணியடித்தது,
    # நிலா+அழகு → நிலாவழகு, சே+அழகு → சேயழகு/சேவழகு (ஏ → both are valid).
    # =====================================================================
    Rule(re.compile(rf"(ி|ீ|ை)([{_V}])")),        # dependent இ/ஈ/ஐ-class matra ஈறு + உயிர் → ய்
                                                       # (fixes வாழை+இலை→வாழையிலை: ழை ends in the
                                                       # "ை" matra, not the independent letter ஐ)
    Rule(re.compile(rf"(ு|ூ)([{_V}])")),          # dependent உ/ஊ-sign ஈறு + உயிர் → வ்
    Rule(re.compile(rf"(ா|ெ|ே|ொ|ோ|ௌ)([{_V}])")), # remaining dependent matras + உயிர் → வ்
                                                       # (fixes நிலா+அழகு→நிலாவழகு: நிலா ends in the
                                                       # "ா" matra; சே+அழகு→சேயழகு: சே ends in "ே")
    Rule(re.compile(rf"(இ|ஈ|ஐ)([{_V}])")),        # independent இ/ஈ/ஐ ஈறு + உயிர் → ய்
    Rule(re.compile(rf"(உ|ஊ)([{_V}])")),          # independent உ/ஊ ஈறு + உயிர் → வ்
    Rule(re.compile(rf"(ஏ)([{_V}])")),            # ஏ ஈறு + உயிர் → ய் அல்லது வ் both valid

    # =====================================================================
    # C) உடல்மேல் உயிர் வந்து ஒன்றுவது இயல்பே — நூற்பா 204 (general rule).
    # ANY word-final consonant + a following vowel-initial word joins
    # automatically without change (தமிழ்+ஆசிரியர் → தமிழாசிரியர்,
    # கடவுள்+அருள் → கடவுளருள், பொருள்+அனைத்தும் → பொருளனைத்தும்).
    # This single general rule subsumes the old per-consonant "கெடுதல்"
    # list (க்/ச்/ட்/த்/ப்/ம்/ய்/ல்/ள்/ர் each followed by a vowel).
    # =====================================================================
    Rule(re.compile(rf"([{_C}]்)([{_V}])")),

    # =====================================================================
    # D) குற்றியலுகரம்/முற்றியலுகரம் + உயிர் (நூற்பா 164)
    # "உயிர் வரின் உக்குறள் மெய் விட்டு ஓடும்." A word-final short/glide-u
    # (குற்றியலுகரம்) loses its consonant before ANY following vowel, not
    # only அ (broader than rule B's ு/ூ+அ pair — higher recall, a shade
    # lower precision since it fires on any u-final word before a vowel).
    # =====================================================================
    #Rule(re.compile(rf"(ு|ூ)([{_V}])")), Already covered in B

    # =====================================================================
    # E) திசைப்பெயர்ப் புணர்ச்சி — நூற்பா 186 (CLOSED lexical set, safe).
    # வடக்கு/தெற்கு/குடக்கு/குணக்கு/கிழக்கு/மேற்கு are a fixed, finite
    # vocabulary of direction words; matching them literally as a
    # nilaimozhi is low-risk because they rarely occur mid-compound
    # for any other reason. வடக்கு+கிழக்கு→வடகிழக்கு, தெற்கு+மேற்கு→
    # தென்மேற்கு, மேற்கு+ஊர்→மேலூர், கிழக்கு+நாடு→கீழ்நாடு.
    # =====================================================================
    # NOTE: intentionally only the full standalone forms — the mutated short
    # roots (வட-, தென்-, குட-, குண-, கீழ்-, மேல்-) that appear in the FUSED
    # compound output (வடகிழக்கு, கீழ்நாடு, மேலூர்...) are dropped: they are
    # short substrings that collide with unrelated words (e.g. "குட" is also
    # just the start of குடை "umbrella"), so detecting the already-mutated
    # compound form reliably needs a dictionary, not a regex.
    Rule(re.compile(r"(வடக்கு|தெற்கு|குடக்கு|குணக்கு|கிழக்கு|மேற்கு)(\S)")),

    # NOTE: மையீற்றுப் பண்புப்பெயர்ப் புணர்ச்சி (நூற்பா 136 — வெண்மை+குடை→
    # வெண்குடை, செம்மை+மலர்→செம்மலர்) is deliberately NOT included. The fused
    # output drops "மை" entirely, so a rule would have to trigger on the bare
    # reduced stem (வெண், செம், வெம் ...) — but those same syllables also just
    # start plenty of unrelated words (செம்பு "copper", வெண்பா "a poem type"),
    # so a regex can't tell the two apart without a dictionary. Skipped rather
    # than shipped broken.

    # =====================================================================
    # G) தனிக்குறில் முன் ஒற்று இரட்டித்தல் (CLOSED, textbook set).
    # "தனிக்குறில் முன் ஒற்று உயிர் வரின் இரட்டும்" — கண்+ஒளி→கண்ணொளி,
    # பண்+ஓசை→பண்ணோசை, மண்+ஓசை→மண்ணோசை. Single-mora CVC roots; kept as
    # a short explicit list rather than a general pattern (a general
    # "short vowel + single consonant" pattern would match the START of
    # almost every Tamil word and be useless).
    # =====================================================================
    Rule(re.compile(rf"(கண்|பண்|மண்|தண்|எண்|வண்)([{_V}])")),

    # =====================================================================
    # H) வேற்றுமை உருபு இணைவு (case-suffix / postposition joins)
    # =====================================================================
    Rule(re.compile(rf"([{_V}])(ஐ)")),                       # 2nd case (accusative) -ஐ
    Rule(re.compile(rf"([{_V}])(உ|க்)கு")),              # 4th case (dative) -க்கு/-உக்கு
    Rule(re.compile(rf"([{_V}])(ஆல்|னால்)")),                # 3rd case (instrumental) -ஆல்/-னால்
    Rule(re.compile(rf"([{_V}])(இல்|அல்)")),                 # 7th case (locative) -இல்/-அல்
    Rule(re.compile(rf"([{_V}])(இடம்|உடன்|முன்|பின்|ஓடு|ஓடே)")),  # postpositions
    Rule(re.compile(rf"([{_V}])(கள்)")),                     # plural marker -கள் (vowel-final noun)
    Rule(re.compile(rf"([{_C}]்)(கள்)")),                    # plural marker -கள் (consonant-final noun,
                                                                  # e.g. புத்தகங்கள்)
    Rule(re.compile(rf"([{_V}])(வர்கள்|வர்)")),               # human-plural agentive -வர்/-வர்கள்

    # =====================================================================
    # I) வினையெச்சம்/துணைவினை இணைவு (verbal participle + auxiliary joins)
    # =====================================================================
    Rule(re.compile(r"(ி|உ|அ)(போ|வா|இரு|உள்|கொள்|விடு|ஆகு)")),  # participle + auxiliary
    Rule(re.compile(r"(த்து|ட்டு)(கொள்|விடு|போ|ஆகு)")),          # -த்து/-ட்டு + auxiliary
    Rule(re.compile(r"(ஆன|என்|உம்)(\S)")),                        # adjectival/relativizer + noun

    # =====================================================================
    # J) எண் + வகுப்பான் இணைவு (numeral + classifier/suffix joins)
    # =====================================================================
    Rule(re.compile(r"([௦-௯0-9]+)(ஆம்(?: நாள்| ஆண்டு)?)")),
    Rule(re.compile(r"([௦-௯0-9]+)(ஐ)")),

    # =====================================================================
    # K) வல்லினம் மிகும் இடங்கள் — CLOSED trigger-word set (safe, high
    # precision because these are exact, unambiguous whole words).
    # அந்த/இந்த + noun, அத்துணை/இத்துணை/எத்துணை + noun, அவ்வகை/இவ்வகை/
    # எவ்வகை + noun, மற்ற/மற்று/மற்றை + noun, அரை/பாதி + noun all trigger
    # a following வல்லின consonant to double (அந்த+பையன்→அந்தப்பையன்).
    # =====================================================================
    Rule(re.compile(r"(அந்த|இந்த|அத்துணை|இத்துணை|எத்துணை|அவ்வகை|இவ்வகை|"
                     r"எவ்வகை|மற்று|மற்றை|மற்ற|அரை|பாதி)(\S)")),
]


# =========================================================================
# AGGRESSIVE / OPTIONAL TIER — real sandhi phenomena, but the same surface
# patterns are also just native in-word spelling (see module docstring).
# Include only if your pipeline can tolerate the extra false positives, or
# if you pair this with a dictionary/exception-word guard.
# =========================================================================
TA_RULES_AGGRESSIVE: List[Rule] = [

    # -------------------------------------------------------------------
    # K2) உடம்படுமெய் ஏற்கனவே எழுதப்பட்ட வடிவம் (glide already spelled out).
    # Real running Tamil text spells the inserted ய்/வ் glide explicitly —
    # மணி+அடித்தது is actually written "மணியடித்தது", not with a raw vowel
    # hiatus — so THIS is the pattern that matches real compound text, not
    # the CORE tier's vowel-vowel rules. High value, but the same shape
    # (matra + ய/வ) also occurs inside ordinary non-compound words (பெரிய
    # "big", செய்ய "to do"), so it's aggressive-tier, not core.
    # -------------------------------------------------------------------
    Rule(re.compile(rf"([{_MATRAS}])(ய|வ)")),

    # -------------------------------------------------------------------
    # L) நாசி + வல்லினம் இனமாதல் (மெய்சந்தி nasal+stop assimilation).
    # Real at genuine compound joins, but ங்க/ஞ்ச/ண்ட/ந்த/ம்ப/ன்ற are
    # also the ordinary native spelling of those clusters inside a
    # single word (அன்பு, தங்கம், இந்த, சிந்தனை, கும்பல், மங்கை, பங்கு,
    # தம்பி all match here with zero sandhi occurring). HIGH RECALL,
    # LOW PRECISION tier.
    # -------------------------------------------------------------------
    Rule(re.compile(r"(ங்)(க)")),
    Rule(re.compile(r"(ஞ்)(ச)")),
    Rule(re.compile(r"(ண்)(ட)")),
    Rule(re.compile(r"(ந்)(த)")),
    Rule(re.compile(r"(ம்)(ப)")),
    Rule(re.compile(r"(ன்)(ற)")),   # fixed: original had (ன்)(ந) — canonical pair is ன்+ற
    Rule(re.compile(r"(ங்|ஞ்|ண்|ந்|ம்|ன்)(க|ச|ட|த|ப|ற)")),  # generic safety net

    # -------------------------------------------------------------------
    # M) மெய் இரட்டித்தல் (gemination/doubling across a boundary). Same
    # caveat as (L): க்க/ச்ச/.../ற்ற/ன்ன occur natively inside ordinary
    # words (பள்ளி, கல்லூரி, நல்ல, எல்லை, வெள்ளை, கொள்ளை) with no sandhi
    # involved at all — this is the exact ambiguity behind the பள்ளி
    # false-split you flagged earlier; a plain regex cannot resolve it
    # without a lexicon.
    # -------------------------------------------------------------------
    # One backreference-based rule covers all 18 consonants doubling across
    # a boundary (க்க, ங்ங, ச்ச, ஞ்ஞ, ட்ட, ண்ண, த்த, ந்ந, ப்ப, ம்ம, ய்ய,
    # ர்ர, ல்ல, வ்வ, ழ்ழ, ள்ள, ற்ற, ன்ன) — e.g. கண்+ஒளி→கண்ணொளி needs ண்ண,
    # which the old hand-picked list of 12 happened to omit.
    Rule(re.compile(rf"([{_C}])்(\1)")),

    # -------------------------------------------------------------------
    # N) திரிதல் (mutation) cues. Same class of ambiguity as (L)/(M).
    # -------------------------------------------------------------------
    Rule(re.compile(r"(ல்)(ச)")),               # ல்+ச mutation environment
    Rule(re.compile(r"(ர்|ற்)(ர)")),             # ர்/ற்+ர mutation environment
    Rule(re.compile(r"(ன்|ண்)(ட|த)")),           # dental/retroflex interplay
]


# =========================================================================
# Whitespace suppression — MUST remain the last element (sandhi.py builds
# TA_PHONOLOGICAL_RULES = TA_RULES[:-1] and computes whitespace boundaries
# separately/correctly instead of via this rule; kept only for drop-in
# shape-compatibility with the old TA_RULES list).
# =========================================================================
_WHITESPACE_RULE = Rule(re.compile(r"(\S)\s+(?=\S)"))

# Default export: same drop-in shape as the original TA_RULES (core +
# aggressive + trailing whitespace rule). For higher precision, use
# TA_RULES_CORE + [_WHITESPACE_RULE] instead in sandhi.py.
TA_RULES: List[Rule] = TA_RULES_CORE + TA_RULES_AGGRESSIVE + [_WHITESPACE_RULE]
