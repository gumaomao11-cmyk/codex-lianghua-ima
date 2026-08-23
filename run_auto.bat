@echo off
setlocal
chcp 65001 >nul
cd /d F:\even-codex\lianghua2
python auto_run.py >> logs\auto_run_cron.log 2>&1
endlocal
