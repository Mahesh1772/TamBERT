@echo off
REM ======================================================================================
REM Creates the conda env and installs the project into it. Run once, before any numbered
REM stage script.
REM
REM Does not use _common.bat: that file checks for an env and an editable install which do
REM not exist yet, which is the whole point of this script.
REM ======================================================================================
setlocal

REM cd to the REPO ROOT, not to bat_files\. requirements.txt and pyproject.toml both live at
REM the root, and the previous version did `cd /d "%~dp0"` and then
REM `pip install -r requirements.txt` -- a file that is not in bat_files\, so the install step
REM could never have succeeded.
cd /d "%~dp0.."
if errorlevel 1 (echo ERROR: could not cd to the repo root. & goto :fail)

if not defined TAMBERT_ENV set "TAMBERT_ENV=tambert_env"

echo Creating conda env "%TAMBERT_ENV%" (python 3.11)...
call conda create -n %TAMBERT_ENV% python=3.11 -y
if errorlevel 1 (
    echo ERROR: conda create failed.
    echo   Env already exists?  remove it with  conda env remove -n %TAMBERT_ENV%
    echo                        or set TAMBERT_ENV to a new name and re-run.
    goto :fail
)

call conda activate %TAMBERT_ENV%
if errorlevel 1 (
    echo ERROR: conda activate failed. Run "conda init cmd.exe" once, reopen the terminal,
    echo        then re-run this script.
    goto :fail
)

echo.
echo Installing pinned dependencies...
REM requirements.txt targets cu126 / torch 2.13.0. A Blackwell / RTX 50-series box needs the
REM cu128 channel instead, which caps torch at 2.11.0 -- change the --extra-index-url and the
REM pin together, and keep the pin at >= 2.6. See the comments in requirements.txt.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    echo   lxml failed to build?  try: conda install -c conda-forge lxml -y
    echo   no matching torch?     your GPU may need cu128 -- see requirements.txt
    goto :fail
)

echo.
echo Installing the project itself (editable)...
REM MANDATORY, and intentionally not expressible in requirements.txt. pyproject.toml sets
REM package-dir = {"" = "scripts"}, which is what makes data_pipeline / tokenizer / mlm / nli
REM importable as top-level packages and paths / training_utils as top-level modules. Every
REM script uses absolute imports with no sys.path manipulation, so without this line nothing
REM under scripts/ runs at all. Omitting it was the most common setup failure.
pip install -e .
if errorlevel 1 (echo ERROR: `pip install -e .` failed. & goto :fail)

echo.
echo Verifying...
python -c "import paths, training_utils; print('imports OK')"
if errorlevel 1 (echo ERROR: installed, but imports still fail. & goto :fail)

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo.
echo Environment "%TAMBERT_ENV%" is ready.
echo.
echo If that printed `cuda False` this is a CPU-only build: stages 03 and 05 cannot run here,
echo both set fp16=True. Stages 01, 02 and 04 are fine on CPU.
echo.
echo Next:  bat_files\01_data_pipeline.bat
endlocal & exit /b 0

:fail
echo.
echo SETUP FAILED - see the error above.
if not defined TAMBERT_NO_PAUSE pause
endlocal & exit /b 1
