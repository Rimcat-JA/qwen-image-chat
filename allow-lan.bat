@echo off
rem Adds a Windows Firewall rule so phones on the same LAN can connect (asks for administrator approval).
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0allow-lan.ps1\"'"
