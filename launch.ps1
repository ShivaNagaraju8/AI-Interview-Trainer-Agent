# AI Interview Trainer — PowerShell launcher
# Double-click this OR run: powershell -ExecutionPolicy Bypass -File launch.ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force -ErrorAction SilentlyContinue
& ".\.venv\Scripts\Activate.ps1"
Write-Host "Launching at http://localhost:8501  (Ctrl+C to stop)" -ForegroundColor Green
.\.venv\Scripts\streamlit.exe run app.py --server.port 8501
