@echo off
REM ======================================================================================
REM Stage 3b - build mlm/<run>/final/ from an existing checkpoint, without retraining.
REM
REM Not part of the normal path: train_roberta.py writes final/ itself at the end of a run.
REM This exists for runs that predate that export step -- 03_sandhi_codepoint_bert is one, so
REM NLI cannot start until this has been run against a recovered checkpoint.
REM
REM Defaults to the run's own best_model_checkpoint, NOT the newest. Early stopping means the
REM newest weights are measurably worse than the best ones. It also NaN-checks the tensors and
REM verifies vocab size against the embedding rows.
REM
REM Runs fine on CPU -- it only reads and rewrites tensors, so no CUDA guard here.
REM
REM Usage:
REM   04_export_backbone.bat
REM   04_export_backbone.bat 03_sandhi_codepoint_bert
REM   04_export_backbone.bat 03_sandhi_codepoint_bert --from checkpoint-548000 --force
REM All arguments are passed straight through to export_backbone.py.
REM ======================================================================================
setlocal

call "%~dp0_common.bat" || goto :fail

echo.
echo Exporting backbone %*
python scripts/mlm/export_backbone.py %* || goto :fail

echo.
echo Export complete. If the script warned that it fell back off best_model_checkpoint, the
echo weights you just exported are NOT the ones any published eval_loss was measured on --
echo do not quote that number for this export.
echo.
echo Next:  bat_files\05_nli_finetune.bat
endlocal & exit /b 0

:fail
echo.
echo EXPORT FAILED - see the error above.
if not defined TAMBERT_NO_PAUSE pause
endlocal & exit /b 1
