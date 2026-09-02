@echo off
setlocal
cd /d "%~dp0"

title AI Hiring Intelligence & Bias Mitigation System

echo.
echo ============================================================
echo   AI Hiring Intelligence ^& Bias Mitigation System
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo.
    echo Expected:
    echo %CD%\.venv\Scripts\python.exe
    echo.
    echo Create it with:
    echo python -m venv .venv
    echo.
    pause
    exit /b 1
)

echo Starting FastAPI server...
echo.
echo Dashboard: http://127.0.0.1:8000
echo Press CTRL+C to stop the server.
echo.

".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
