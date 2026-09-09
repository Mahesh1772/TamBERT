@echo off
REM ======================================================================================
REM Stage 3 - MLM pre-training. RobertaForMaskedLM, 6L/768H, 32k vocab, on the
REM sandhi-marked corpus with the 03_sandhi_codepoint_bert tokenizer.
REM
REM GPU box only, and measured in days rather than hours. Resumes automatically from a
REM validated checkpoint, so re-running after an interruption is the intended way to use it.
REM
REM BEFORE THE FIRST RUN, check where output_dir points. The first MLM run trained inside a
REM OneDrive-synced folder and lost five of its six checkpoints -- sync dehydrates or fails on
REM the ~270 MB weight and ~540 MB optimizer files, and it fails silently. output_dir must be
REM a local, non-synced path.
REM ======================================================================================
setlocal

call "%~dp0_common.bat" || goto :fail

REM --- CUDA guard -----------------------------------------------------------------------
REM train_roberta.py sets fp16=True, which cannot work on a CPU-only torch build. Without
REM this check the run dies partway through setup with a less obvious message. The dev box in
REM this project is deliberately CPU-only, so tripping this is expected there, not a bug.
if defined TAMBERT_ALLOW_CPU goto :cuda_ok
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: no CUDA device visible, and train_roberta.py sets fp16=True.
    echo   This stage belongs on the training box, not the CPU-only dev box.
    echo   To run anyway, remove fp16=True from the script first, then
    echo     set TAMBERT_ALLOW_CPU=1
    goto :fail
)
:cuda_ok

python -c "import torch; print('[gpu]', torch.cuda.get_device_name(0))" 2>nul

echo.
echo Starting MLM pre-training (resumes from a validated checkpoint if one exists)...
REM resolve_resume_checkpoint requires BOTH weights and optimizer.pt before it will resume.
REM HF's own get_last_checkpoint only pattern-matches the directory name, so it happily
REM returns a gutted checkpoint and then fails deep inside Trainer; and a weights-only resume
REM silently reinitializes the optimizer moments and restarts the LR schedule from step 0,
REM which is a quality regression rather than an error. Watch stdout for the loud warning that
REM the newest checkpoint had to be skipped.
python scripts/mlm/train_roberta.py || goto :fail

echo.
echo Stage 3 complete. final_metrics.json and the full log history are at the run root,
echo and the weights-only export the next stage consumes is in mlm\^<run^>\final\.
echo.
echo Report eval_loss, not the logged `loss` column -- gradient_accumulation_steps=4 inflates
echo the latter 4x. Then write up the run in mlm\^<run^>\README.md.
echo.
echo Next:  bat_files\05_nli_finetune.bat
echo        (04_export_backbone.bat is only needed if this run produced no final\)
endlocal & exit /b 0

:fail
echo.
echo STAGE 3 FAILED - see the error above.
if not defined TAMBERT_NO_PAUSE pause
endlocal & exit /b 1
