@echo off
REM ============================================================
REM Yard Monitor — Windows launcher
REM   - activates the local venv
REM   - starts the FastAPI server (cameras + dashboard)
REM   - all output stays in this terminal window
REM ============================================================

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Run:  python -m venv .venv  and pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo  Yard Monitor
echo  Dashboard: http://localhost:8000
echo  Press Ctrl-C to stop.
echo.
python main.py
