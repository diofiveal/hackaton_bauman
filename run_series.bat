@echo off
setlocal
cd /d "%~dp0"
python series_runner.py %*
set "STATUS=%ERRORLEVEL%"
if not "%STATUS%"=="0" echo ERROR: Series run failed with exit code %STATUS%.
exit /b %STATUS%
