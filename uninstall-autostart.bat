@echo off
rem Remove the automatic start and stop the running app.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall-autostart.ps1"
pause
