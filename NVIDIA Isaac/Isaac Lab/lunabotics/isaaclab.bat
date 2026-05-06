@echo off
setlocal

rem ── Lunabotics Isaac Lab project wrapper (Windows) ─────────────────────────
rem Reads ISAACLAB_PATH from isaaclab_path.cfg (next to this script) and
rem forwards all arguments to the real isaaclab.bat in that installation.
rem
rem Usage:  isaaclab.bat -p scripts\rsl_rl\train.py --task Template-Lunabotics-Direct-v0

set "SCRIPT_DIR=%~dp0"
set "CFG_FILE=%SCRIPT_DIR%isaaclab_path.cfg"

if not exist "%CFG_FILE%" (
    echo ERROR: %CFG_FILE% not found.
    echo Copy isaaclab_path.cfg.example to isaaclab_path.cfg and set ISAACLAB_PATH.
    exit /b 1
)

rem Parse ISAACLAB_PATH=value from cfg file
for /f "usebackq tokens=1,* delims==" %%A in ("%CFG_FILE%") do (
    if "%%A"=="ISAACLAB_PATH" set "ISAACLAB_PATH=%%B"
)

if not defined ISAACLAB_PATH (
    echo ERROR: ISAACLAB_PATH not set in %CFG_FILE%
    exit /b 1
)

if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
    echo ERROR: isaaclab.bat not found at %ISAACLAB_PATH%
    echo Check that ISAACLAB_PATH in isaaclab_path.cfg is correct.
    exit /b 1
)

call "%ISAACLAB_PATH%\isaaclab.bat" %*
