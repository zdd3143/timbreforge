@echo off
setlocal
cd /d "%~dp0"
python -m timbreforge.launcher %*
if errorlevel 1 pause
