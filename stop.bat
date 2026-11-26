@echo off
rem Stops the background image engine (sd-server) started by start.bat.
taskkill /IM sd-server.exe /F
pause
