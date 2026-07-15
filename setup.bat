@echo off
REM ╔══════════════════════════════════════════════════════════════╗
REM ║  AI Interview Trainer Agent — Windows Setup Script          ║
REM ║  Run this ONCE after installing Python from python.org      ║
REM ╚══════════════════════════════════════════════════════════════╝

echo.
echo  Checking Python installation...
echo.

REM ── Try common Python locations ───────────────────────────────
set PYTHON=
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Program Files\Python313\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
) do (
    if exist %%P (
        set PYTHON=%%P
        goto :found
    )
)

REM ── Python not found ──────────────────────────────────────────
echo  [ERROR] Python not found.
echo.
echo  Please install Python 3.10 or higher:
echo    1. Go to  https://www.python.org/downloads/
echo    2. Download the Windows installer (64-bit recommended)
echo    3. During install, CHECK "Add Python to PATH"
echo    4. Re-run this script after installation
echo.
pause
exit /b 1

:found
echo  [OK] Python found: %PYTHON%
%PYTHON% --version

echo.
echo  Creating virtual environment...
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo  [OK] Virtual environment created in .venv\

echo.
echo  Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo  Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo  Installing dependencies (this may take 3-5 minutes)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Some packages failed to install.
    echo  Try running manually: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo  Setting up .env file...
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo  [OK] Created .env — please edit it with your IBM credentials
) else (
    echo  [OK] .env already exists
)

echo.
echo  ============================================================
echo   Setup Complete!
echo  ============================================================
echo.
echo   NEXT STEPS:
echo   1. Edit .env and add your IBM watsonx.ai credentials
echo      Get API key:    https://cloud.ibm.com/iam/apikeys
echo      Get Project ID: https://dataplatform.cloud.ibm.com
echo.
echo   2. Run the app:
echo      run_app.bat
echo      -- OR --
echo      .venv\Scripts\activate.bat
echo      streamlit run app.py
echo.
pause
