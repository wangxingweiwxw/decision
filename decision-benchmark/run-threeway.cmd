@echo off
cd /d "%~dp0"
python benchmark.py --env-override run --providers bailian,microsoft,qwen --rounds 3 --warmup 2 --concurrency 1
pause
