@echo off
REM ======================================================================================
REM Stage 4 - NLI fine-tuning. IndicXNLI Tamil as a cross-encoder on top of the MLM
REM backbone: RobertaForSequenceClassification with a fresh 3-way head.
REM
REM GPU box only (fp16=True). Auto-downloads IndicXNLI as a zip via gdown on first run and
REM skips the download if all three splits are already extracted.
REM
REM Consumes mlm/<run>/final/, not a checkpoint. If that directory is missing the script
REM raises immediately and names the fix -- run 04_export_backbone.bat. No need to duplicate
REM that check here; the script's own error is clearer than anything batch can produce.
REM ======================================================================================
setlocal

call "%~dp0_common.bat" || goto :fail

REM --- CUDA guard -----------------------------------------------------------------------
REM Same reason as stage 03: train_nli.py sets fp16=True.
if defined TAMBERT_ALLOW_CPU goto :cuda_ok
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: no CUDA device visible, and train_nli.py sets fp16=True.
    echo   This stage belongs on the training box, not the CPU-only dev box.
    echo   To run anyway, remove fp16=True from the script first, then
    echo     set TAMBERT_ALLOW_CPU=1
    goto :fail
)
:cuda_ok

python -c "import torch; print('[gpu]', torch.cuda.get_device_name(0))" 2>nul

echo.
echo Starting NLI fine-tuning...
REM The "newly initialized classifier weights" warning transformers prints at load is
REM expected -- the lm_head is dropped and a fresh 3-way head replaces it.
REM
REM Watch the two lines the script prints before training starts:
REM   * "Input preprocessing for <run>" -- must name the backbone tokenizer's own marking
REM     scheme. Feeding raw IndicXNLI to a sandhi-trained tokenizer strands 1,253 of the
REM     32,000 embedding rows and pushes fertility 1.3412 -> 1.4141, with no error either way.
REM   * the per-split truncation rate at max_length=128, so that budget stays honest.
python scripts/nli/train_nli.py || goto :fail

echo.
echo Stage 4 complete. Test accuracy, macro-F1 and per-class F1 are in the run root's
echo final_metrics.json, with the weights-only export in nli\^<run^>\final\.
echo.
echo That export is what STS fine-tuning would consume -- but stage 5 is blocked on data,
echo not code: there is no good-quality Tamil-Tamil STS set. See the README.
endlocal & exit /b 0

:fail
echo.
echo STAGE 4 FAILED - see the error above.
if not defined TAMBERT_NO_PAUSE pause
endlocal & exit /b 1
