@echo off
set YM_USE_GPU=1
set YM_DEBUG_CROPS=1
cd /d "C:\Users\Nihal Bin Riyas\OneDrive\Desktop\yard-monitor"
"C:\Users\Nihal Bin Riyas\OneDrive\Desktop\yard-monitor\.venv\Scripts\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
