@echo off
REM AI Interview Trainer Agent — Launch Script
echo.
echo  Starting AI Interview Trainer Agent...
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo  [ERROR] .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

if not exist ".env" (
    echo  [WARN] .env not configured - running in Demo Mode
    echo  Copy .env.example to .env and add IBM credentials for full AI features.
    echo.
)

echo  Opening http://localhost:8501
echo  Press Ctrl+C to stop the server.
echo.
.venv\Scripts\streamlit.exe run app.py --server.port 8501
pause
