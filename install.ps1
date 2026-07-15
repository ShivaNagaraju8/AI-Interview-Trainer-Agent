# AI Interview Trainer Agent — PowerShell Setup
# Run: Right-click -> "Run with PowerShell"  OR  .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI Interview Trainer Agent — Setup" -ForegroundColor Cyan
Write-Host "  Powered by IBM watsonx.ai + IBM Granite" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Find Python ─────────────────────────────────────────────────────
Write-Host "Step 1: Locating Python..." -ForegroundColor Yellow

$pythonCandidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Python313\python.exe",
    "C:\Python312\python.exe",
    "C:\Python311\python.exe",
    "C:\Python310\python.exe",
    "C:\Program Files\Python313\python.exe",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe"
)

$pythonExe = $null
foreach ($p in $pythonCandidates) {
    if (Test-Path $p) {
        $pythonExe = $p
        break
    }
}

# Fallback: check if python is on PATH (e.g. conda)
if (-not $pythonExe) {
    try {
        $fromPath = (Get-Command python -ErrorAction Stop).Source
        if ($fromPath -notmatch "WindowsApps") {
            $pythonExe = $fromPath
        }
    } catch {}
}

if (-not $pythonExe) {
    Write-Host ""
    Write-Host "  ERROR: Python not found on this machine." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please install Python 3.10+ first:" -ForegroundColor White
    Write-Host "  1. Open: https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host "  2. Download 'Windows installer (64-bit)'" -ForegroundColor White
    Write-Host "  3. Run installer and CHECK 'Add Python to PATH'" -ForegroundColor White
    Write-Host "  4. Re-run this script" -ForegroundColor White
    Write-Host ""
    Write-Host "  Quick install with winget (if available):" -ForegroundColor Yellow
    Write-Host "  winget install Python.Python.3.12" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$version = & $pythonExe --version 2>&1
Write-Host "  OK: $pythonExe ($version)" -ForegroundColor Green

# ── Step 2: Create virtual environment ──────────────────────────────────────
Write-Host ""
Write-Host "Step 2: Creating virtual environment (.venv)..." -ForegroundColor Yellow

if (Test-Path ".venv") {
    Write-Host "  OK: .venv already exists, skipping creation." -ForegroundColor Green
} else {
    & $pythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create .venv" -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: .venv created" -ForegroundColor Green
}

$pipExe = ".\.venv\Scripts\pip.exe"
$pyVenv = ".\.venv\Scripts\python.exe"

# ── Step 3: Upgrade pip ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 3: Upgrading pip..." -ForegroundColor Yellow
& $pyVenv -m pip install --upgrade pip --quiet
Write-Host "  OK: pip upgraded" -ForegroundColor Green

# ── Step 4: Install dependencies ────────────────────────────────────────────
Write-Host ""
Write-Host "Step 4: Installing packages from requirements.txt..." -ForegroundColor Yellow
Write-Host "  (This may take 3-5 minutes on first run)" -ForegroundColor Gray

& $pipExe install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  ERROR: Some packages failed. Try manually:" -ForegroundColor Red
    Write-Host "  .\.venv\Scripts\pip.exe install -r requirements.txt" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  OK: All packages installed" -ForegroundColor Green

# ── Step 5: Create .env ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 5: Configuring environment..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  OK: Created .env from .env.example" -ForegroundColor Green
    Write-Host "  --> Edit .env and add your IBM Cloud credentials" -ForegroundColor Yellow
} else {
    Write-Host "  OK: .env already exists" -ForegroundColor Green
}

# ── Done ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Edit .env and add your IBM watsonx.ai credentials:" -ForegroundColor White
Write-Host "     WATSONX_API_KEY   -> https://cloud.ibm.com/iam/apikeys" -ForegroundColor Cyan
Write-Host "     WATSONX_PROJECT_ID -> https://dataplatform.cloud.ibm.com" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Run the app (pick one):" -ForegroundColor White
Write-Host "     Double-click:  run_app.bat" -ForegroundColor Cyan
Write-Host "     PowerShell:    .\.venv\Scripts\streamlit.exe run app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  App runs at: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
