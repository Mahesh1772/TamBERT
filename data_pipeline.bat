@echo off
cd /d "%~dp0"

echo Running the data pipeline...
call conda activate tambert_env
if errorlevel 1 (
    echo ERROR: Failed to activate tambert_env. Run setup_env.bat first.
    exit /b 1
)

echo Extracting data from Wikipedia...
python scripts/data_pipeline/01_extract_wiki.py
if errorlevel 1 (echo ERROR: Wikipedia extraction failed. & exit /b 1)
echo Wikipedia data extraction complete.

echo Extracting data from Common Crawl...
python scripts/data_pipeline/02_filter_cc100.py
if errorlevel 1 (echo ERROR: CC100 filtering failed. & exit /b 1)
echo Common Crawl data filtering complete.

echo Extracting data from Project Madurai...
python scripts/data_pipeline/03_scrape_madurai.py
if errorlevel 1 (echo ERROR: Project Madurai scraping failed. & exit /b 1)
echo Project Madurai data extraction complete.

echo Cleaning the extracted data...
python scripts/data_pipeline/04_clean_data.py
if errorlevel 1 (echo ERROR: Data cleaning failed. & exit /b 1)
echo Data cleaning complete.

echo Deduplicating and Merging the cleaned data...
python scripts/data_pipeline/05_merge_corpus.py
if errorlevel 1 (echo ERROR: Merge/dedup failed. & exit /b 1)
echo Deduplication and merging complete.

echo Calculating metrics of cleaned data...
python scripts/data_pipeline/06_calculate_corpus_metrics.py
if errorlevel 1 (echo ERROR: Metrics calculation failed. & exit /b 1)
echo Metrics calculation complete.

echo Pipeline finished successfully.