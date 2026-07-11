@echo off
cd /d "%~dp0"

echo Setting up the environment...
call conda create -n tambert_env python=3.11 -y
if errorlevel 1 (
    echo ERROR: conda create failed.
    exit /b 1
)

call conda activate tambert_env
if errorlevel 1 (
    echo ERROR: conda activate failed. Run "conda init cmd.exe" once and reopen terminal.
    exit /b 1
)

echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. If lxml fails to build, try: conda install -c conda-forge lxml -y
    exit /b 1
)

echo Environment setup complete.