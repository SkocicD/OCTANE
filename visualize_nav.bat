@echo off
setlocal enabledelayedexpansion

set ARGS=

:parse
if "%~1"=="" goto run
if /i "%~1"=="--checkpoint" set ARGS=%ARGS% --checkpoint "%~2"& shift & shift & goto parse
if /i "%~1"=="--seed"       set ARGS=%ARGS% --seed "%~2"&       shift & shift & goto parse
if /i "%~1"=="--phase"      set ARGS=%ARGS% --phase "%~2"&      shift & shift & goto parse
if /i "%~1"=="--port"       set ARGS=%ARGS% --port "%~2"&       shift & shift & goto parse
shift
goto parse

:run
echo Starting navigation visualizer...
python training_nav/visualize.py%ARGS%
if errorlevel 1 (
    echo.
    echo Error -- make sure dependencies are installed: setup_nav_training.bat
    pause
)
