@echo off
cd /d "%~dp0"
REM 四方对比：百炼 / 微软 Decision-1 / Qwen3.5-9B / TypeSafe Jev
REM Jev 需要 TYPESAFE_API_KEY（waitlist 批准后，仅美国西海岸可用）。
python benchmark.py --env-override run --providers bailian,microsoft,qwen,jev --rounds 3 --warmup 2 --concurrency 1
pause
