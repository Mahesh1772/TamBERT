@echo off
cd /d "%~dp0"

echo Running the data pipeline...
call conda activate tambert
if errorlevel 1 (
echo ERROR: Failed to activate tambert. Run setup_env.bat first.
pause
exit /b 1
)

echo Generating sandhi split text files...
python scripts/tokenizer/sandhi_precompute.py
if errorlevel 1 (echo ERROR: Sandhi split generation failed. & pause & exit /b 1)
echo Sandhi split text files generated successfully.

echo Generating sandhi split grapheme files...
python scripts/tokenizer/grapheme_precompute.py
if errorlevel 1 (echo ERROR: Sandhi split grapheme generation failed. & pause & exit /b 1)
echo Sandhi split grapheme files generated successfully.

echo Running BPE tokenizer training...
python scripts/tokenizer/bpe/train_bpe_tokenizer.py
if errorlevel 1 (echo ERROR: BPE tokenizer training failed. & pause & exit /b 1)
echo BPE tokenizer training completed successfully.

pause