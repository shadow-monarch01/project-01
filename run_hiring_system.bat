@echo off
title AI Hiring System - Bias Detection & Mitigation
cd /d "%~dp0"

echo ======================================================================
echo   AI Hiring System - Bias Detection ^& Explanation Faithfulness
echo ======================================================================
echo.
echo Locating Python environment...

set PYTHON_EXE=python

if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo Using local .venv
) else if exist "..\llm_bias_detection_project\.venv\Scripts\python.exe" (
    set PYTHON_EXE=..\llm_bias_detection_project\.venv\Scripts\python.exe
    echo Using desktop virtual environment
) else if exist "C:\Users\Ashok kumar S\Desktop\llm_bias_detection_project\.venv\Scripts\python.exe" (
    set PYTHON_EXE=C:\Users\Ashok kumar S\Desktop\llm_bias_detection_project\.venv\Scripts\python.exe
    echo Using desktop virtual environment
)

echo Starting FastAPI Web Server on http://127.0.0.1:8000 ...
echo Opening web application in your browser...
echo.

start http://127.0.0.1:8000

"%PYTHON_EXE%" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

echo.
echo Hiring System server stopped.
pause
