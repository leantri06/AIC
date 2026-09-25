@echo off
title PEGASUS - SigLIP2
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if not defined AIC_DATA_ROOT set "AIC_DATA_ROOT=%~dp0..\data"
if not defined HF_HOME set "HF_HOME=%AIC_DATA_ROOT%\.cache\huggingface"
if not defined TORCH_HOME set "TORCH_HOME=%AIC_DATA_ROOT%\.cache\torch"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run src/aic2026/app/ui.py --server.port 8501 --server.address 127.0.0.1 --server.headless false --server.fileWatcherType none
) else (
    echo Missing .venv. See README.md for installation instructions.
    pause
    exit /b 1
)
if errorlevel 1 pause
