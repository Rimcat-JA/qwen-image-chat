@echo off
rem Register Qwen Image Chat to start automatically at Windows logon (no admin needed).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-autostart.ps1"
pause
