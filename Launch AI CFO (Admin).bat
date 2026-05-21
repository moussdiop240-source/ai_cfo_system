@echo off
setlocal enabledelayedexpansion

title AI CFO System — Admin Launcher

:: ── Self-elevate via UAC if not already running as administrator ──
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command ^
      "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

:: ── Already elevated — hand off to the main launcher ─────────────
cd /d "%~dp0"

if not exist "Launch AI CFO.bat" (
    echo.
    echo  ERROR: "Launch AI CFO.bat" not found in:
    echo  %~dp0
    echo.
    echo  Make sure this file is in the ai_cfo_system project folder.
    echo.
    pause
    exit /b 1
)

call "Launch AI CFO.bat"
