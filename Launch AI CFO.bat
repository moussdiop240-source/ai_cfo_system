@echo off
setlocal enabledelayedexpansion

title AI CFO System — Launcher

echo.
echo  ============================================================
echo    AI CFO System  ^|  Multi-Agent Financial Intelligence
echo  ============================================================
echo.

:: ── Change to the directory where this .bat lives ────────────
cd /d "%~dp0"


:: ── STEP 1: Python check ──────────────────────────────────────
echo  [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python was not found on this machine.
    echo.
    echo  Please install Python 3.10 or higher:
    echo    https://www.python.org/downloads/
    echo.
    echo  During installation, tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   OK  Python !PYVER!
echo.


:: ── STEP 2: Virtual environment ───────────────────────────────
echo  [2/4] Setting up virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo   Creating .venv — this takes ~30 seconds on first run...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: Could not create a virtual environment.
        echo  Make sure the "venv" module is available:
        echo    python -m ensurepip --upgrade
        echo.
        pause
        exit /b 1
    )
    echo   OK  .venv created.
) else (
    echo   OK  .venv already exists.
)
echo.


:: ── STEP 3: Dependencies ──────────────────────────────────────
echo  [3/4] Checking dependencies...

if not exist "requirements.txt" (
    echo.
    echo  ERROR: requirements.txt not found.
    echo  Make sure you are running this file from the project folder.
    echo.
    pause
    exit /b 1
)

:: Use streamlit presence as a fast proxy for "deps installed"
.venv\Scripts\python.exe -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing packages — this can take a few minutes on first run...
    echo.
    .venv\Scripts\pip.exe install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: Dependency installation failed.
        echo  Check your internet connection, then try again.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo   OK  All packages installed.
) else (
    echo   OK  Dependencies already installed.
)
echo.


:: ── STEP 4: Launch ────────────────────────────────────────────
echo  [4/4] Starting AI CFO System...
echo.
echo   Your browser will open automatically.
echo   Press Ctrl+C in this window to stop the server.
echo.
echo  ============================================================
echo.

.venv\Scripts\streamlit.exe run streamlit_app.py ^
    --server.headless false ^
    --browser.gatherUsageStats false

if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Streamlit exited with an error ^(see above^).
    echo.
    pause
    exit /b 1
)

endlocal
