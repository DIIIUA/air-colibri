@echo off
setlocal
rem AirColibri: aircolibri.cmd <doctor|ask|chat|serve|discover> [options]
set "PYTHONDONTWRITEBYTECODE=1"
set "PY=python"
if defined AIRCOLIBRI_PYTHON (
    set "PY=%AIRCOLIBRI_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
)
"%PY%" -X utf8 -u "%~dp0aircolibri.py" %*
exit /b %ERRORLEVEL%
