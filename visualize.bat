@echo off
setlocal enabledelayedexpansion

set ARGS=

:parse
if "%~1"=="" goto run
if /i "%~1"=="--gt-only"   set ARGS=%ARGS% --gt-only&  shift & goto parse
if /i "%~1"=="--episode"   set ARGS=%ARGS% --episode "%~2"& shift & shift & goto parse
if /i "%~1"=="--checkpoint" set ARGS=%ARGS% --checkpoint "%~2"& shift & shift & goto parse
shift
goto parse

:run
echo Starting terrain visualizer...
python training/visualize.py%ARGS%
if errorlevel 1 (
    echo.
    echo Error — make sure dependencies are installed: setup_training.bat
    pause
)
