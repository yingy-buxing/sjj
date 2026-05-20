@echo off
set "APP_DIR=%~dp0"
set "PYTHONW=%APP_DIR%.venv\Scripts\pythonw.exe"
set "SCRIPT=%APP_DIR%person_monitor.py"

powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%PYTHONW%' -ArgumentList '\"%SCRIPT%\"' -WorkingDirectory '%APP_DIR%' -Verb RunAs"
