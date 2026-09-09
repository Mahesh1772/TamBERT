@echo off
REM ======================================================================================
REM Stage 2 - corpus variants, then all 12 tokenizers.
REM
REM Replaces sandhi_grapheme_generator.bat, which activated an env name setup_env.bat never
REM creates, pointed at pre-refactor module paths (scripts/tokenizer/*_precompute.py rather
REM than scripts/tokenizer/core/), and trained only the BPE family.
REM
REM Two ordering constraints, both load-bearing:
REM   * Both precomputes must finish before any tokenizer training -- the trainers read the
REM     marked corpora as plain input files and will not generate them.
REM   * Sandhi must run before grapheme. grapheme_precompute reads the sandhi-marked files to
REM     build the stacked *_sandhi_grapheme_marked variant, and inside the marking pipeline
REM     sandhi has to be applied first regardless: run the other way round, the sandhi regexes
REM     would be matching phonological rules against Private-Use-Area placeholders.
REM ======================================================================================
setlocal

call "%~dp0_common.bat" || goto :fail

echo.
echo [1/6] Sandhi precompute (train + test -^> *_sandhi_marked)...
python scripts/tokenizer/core/sandhi_precompute.py || goto :fail

echo.
echo [2/6] Grapheme precompute (-^> *_grapheme_marked, *_sandhi_grapheme_marked)...
REM Also writes tokenizer/grapheme_placeholder_map.json, the cluster -> PUA codepoint table
REM that relabel_vocab_after_save needs later.
python scripts/tokenizer/core/grapheme_precompute.py || goto :fail

echo.
echo [3/6] Training WordPiece family (4 variants)...
REM Each train_*_tokenizer.py is a ~20-line argparse shim over
REM tokenizer/core/pipeline.py, driven by that folder's config.yaml. To retrain a single
REM variant instead of all four:
REM   python scripts/tokenizer/bert/train_bert_tokenizer.py --only 03_sandhi_codepoint_bert
python scripts/tokenizer/bert/train_bert_tokenizer.py || goto :fail

echo.
echo [4/6] Training BPE family (4 variants)...
python scripts/tokenizer/bpe/train_bpe_tokenizer.py || goto :fail

echo.
echo [5/6] Training Unigram family (4 variants)...
REM The only family with relabel_vocab_after_save enabled: BPE and WordPiece merge lists
REM reference vocab entries and would need updating in sync, which was never verified.
python scripts/tokenizer/unigram/train_unigram_tokenizer.py || goto :fail

echo.
echo [6/6] Cross-variant metrics + mBERT baseline...
REM Downloads bert-base-multilingual-cased to run the identical metric function on the
REM identical raw test file. Slow: it encodes the full 3.4M-line test set per variant.
python scripts/tokenizer/core/tokenizer_stats.py || goto :fail

echo.
echo Stage 2 complete. Fertility and OOV per variant are in
echo tokenizer\^<variant^>\metrics.json, with the cross-variant plots and CSVs in
echo tokenizer\tokenizer_statistics\.
echo.
echo Next:  bat_files\03_mlm_pretrain.bat  (GPU box only)
endlocal & exit /b 0

:fail
echo.
echo STAGE 2 FAILED - see the error above.
if not defined TAMBERT_NO_PAUSE pause
endlocal & exit /b 1
