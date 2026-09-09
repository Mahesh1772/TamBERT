@echo off
REM ======================================================================================
REM Shared preamble for every stage script in this folder: repo root, conda env, sanity
REM checks. Callers use:  call "%~dp0_common.bat" || goto :fail
REM
REM It exists because the three original scripts here each carried their own copy of this
REM block and had already drifted apart -- sandhi_grapheme_generator.bat activated an env
REM name that setup_env.bat never creates. One copy cannot drift.
REM
REM Deliberately NOT wrapped in setlocal. `conda activate` works by rewriting PATH and the
REM CONDA_* vars, and setlocal would discard every one of them the instant this file
REM returned, dropping the caller back onto system python with no error.
REM ======================================================================================

REM --- repo root ------------------------------------------------------------------------
REM The stage scripts invoke `python scripts/<pkg>/<file>.py` -- paths relative to the repo
REM ROOT. The original scripts did `cd /d "%~dp0"`, which parks the CWD in bat_files\ where
REM scripts\ does not exist, so not one of them could ever have run. "%~dp0.." is one level
REM up. (Paths itself is CWD-independent -- it derives PROJECT_ROOT from __file__ -- so this
REM cd is about locating the .py files, not about where their outputs land.)
cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: could not cd to the repo root from "%~dp0".
    exit /b 1
)

REM --- conda env ------------------------------------------------------------------------
REM Overridable because this project runs on two machines with different env names: the
REM training box has tambert_env (what setup_env.bat creates), the CPU-only dev box has
REM tambert. `set TAMBERT_ENV=tambert` before calling to switch.
if not defined TAMBERT_ENV set "TAMBERT_ENV=tambert_env"

REM `call` is required. conda on Windows is conda.bat, and invoking a .bat from a .bat
REM without `call` transfers control permanently -- the rest of this file would never run.
call conda activate %TAMBERT_ENV%
if errorlevel 1 (
    echo ERROR: could not activate conda env "%TAMBERT_ENV%".
    echo   env not created yet   -^> run bat_files\setup_env.bat
    echo   conda not on PATH     -^> run "conda init cmd.exe" once, then reopen the terminal
    echo   different env name    -^> set TAMBERT_ENV=your_env_name
    exit /b 1
)

REM --- editable install -----------------------------------------------------------------
REM pyproject.toml declares scripts/ as the package root, and that is the only reason
REM `from paths import Paths` resolves anywhere in this project -- no script does sys.path
REM surgery. Without `pip install -e .` every stage below dies on ModuleNotFoundError, but
REM only after loading torch, so the real cause is buried under a slow traceback. Importing
REM paths is a clean probe: the module has no import-time side effects, it only computes
REM PROJECT_ROOT and defines a dataclass.
python -c "import paths" 2>nul
if errorlevel 1 (
    echo ERROR: this project is not installed into "%TAMBERT_ENV%".
    echo   run:  pip install -e .
    echo   ^(it is required, and requirements.txt does not cover it -- see CLAUDE.md^)
    exit /b 1
)

echo [env] %TAMBERT_ENV%  ^|  [cwd] %CD%
exit /b 0
