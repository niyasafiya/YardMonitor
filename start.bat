@echo off
REM ============================================================
REM Yard Monitor — Windows launcher
REM   - activates the local venv
REM   - starts the FastAPI server (cameras + dashboard)
REM   - opens the dashboard in your default browser
REM ============================================================

setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run:  python -m venv venv  and pip install -r requirements.txt
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"

REM Open the dashboard ~3 seconds after the server starts
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:8000"

echo Starting Yard Monitor on http://localhost:8000
echo Press Ctrl-C to stop.
python main.py
