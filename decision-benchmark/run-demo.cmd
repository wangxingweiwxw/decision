@echo off
cd /d "%~dp0"
python benchmark.py run --mock
pause
