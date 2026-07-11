@echo off
echo Setting up the environment...
call conda create -n tambert_env python=3.11 -y
call conda activate tambert_env
echo Installing required packages...
pip install -r requirements.txt
echo Environment setup complete.