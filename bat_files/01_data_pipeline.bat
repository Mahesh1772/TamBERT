@echo off
REM ======================================================================================
REM Stage 1 - build the corpus.
REM
REM Strictly ordered: each script reads the previous one's output, and none of them take CLI
REM arguments (every path comes from Paths). Long-running and network-heavy -- 01 and 02
REM auto-download large archives (the tawiki dump, CC-100 ta.txt.xz) and 03 scrapes
REM projectmadurai.org live, so expect hours and don't run it on a metered connection.
REM
REM Re-running is safe; the download steps skip archives already on disk.
REM ======================================================================================
setlocal

call "%~dp0_common.bat" || goto :fail

echo.
echo [1/6] Extracting Tamil Wikipedia...
python scripts/data_pipeline/01_extract_wiki.py || goto :fail

echo.
echo [2/6] Filtering CC-100 Tamil...
python scripts/data_pipeline/02_filter_cc100.py || goto :fail

echo.
echo [3/6] Scraping Project Madurai...
python scripts/data_pipeline/03_scrape_madurai.py || goto :fail

echo.
echo [4/6] Cleaning (content-ratio filter)...
REM Splits each source into *_above_threshold / *_below_threshold at a 0.3 real-content
REM ratio. Below-threshold lines are written out for audit, not discarded.
python scripts/data_pipeline/04_clean_data.py || goto :fail

echo.
echo [5/6] Merging + blake2b dedup + 90/10 split...
REM Seed 42. The 10%% test split is permanent -- it is the held-out set every tokenizer
REM fertility number and every MLM eval_loss in this project is measured on.
python scripts/data_pipeline/05_merge_corpus.py || goto :fail

echo.
echo [6/6] Corpus + hygiene metrics...
python scripts/data_pipeline/06_calculate_corpus_metrics.py || goto :fail

echo.
echo Stage 1 complete. Check data\metrics\corpus_metrics_summary.csv and
echo hygiene_metrics_summary.csv -- there is no test suite, those files are the check.
echo.
echo Next:  bat_files\02_tokenizers.bat
endlocal & exit /b 0

:fail
echo.
echo STAGE 1 FAILED - see the error above.
if not defined TAMBERT_NO_PAUSE pause
endlocal & exit /b 1
