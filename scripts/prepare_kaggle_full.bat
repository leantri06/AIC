@echo off
"%~dp0..\.venv\Scripts\python.exe" -u "%~dp0prepare_kaggle_batch2.py" --output "%~dp0..\..\data\batches\batch2\kaggle-full" --shard-frames 2048 --group-gib 8
exit /b %errorlevel%
