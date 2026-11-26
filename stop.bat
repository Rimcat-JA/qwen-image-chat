@echo off
rem Stops Qwen Image Chat: the chat app (port 7860, incl. the Telegram bot) and the image engine (sd-server).
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue; foreach ($x in $c) { Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue }"
taskkill /IM sd-server.exe /F >nul 2>&1
echo Qwen Image Chat stopped.
if not "%1"=="nopause" pause
