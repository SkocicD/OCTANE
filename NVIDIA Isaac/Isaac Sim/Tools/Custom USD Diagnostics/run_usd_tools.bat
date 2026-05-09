@echo off
setlocal enabledelayedexpansion

REM ================================================================
REM  USD Isaac Lab Compliance + Fix Pipeline
REM
REM  Runs three tools in sequence:
REM    1. check_usd_compliance  — read-only audit (PASS/WARN/FAIL)
REM    2. disable_nonwheel_collision — disables arm/non-drive collision
REM    3. lock_arm_joints       — replaces arm drives with position-hold
REM
REM  Usage:
REM    run_usd_tools.bat "C:\path\to\robot.usd"
REM
REM  Edit the CONFIG section below once per robot.
REM ================================================================

REM ----------------------------------------------------------------
REM  CONFIG — edit these for your robot
REM ----------------------------------------------------------------

REM  Path to isaaclab.bat
set ISAACLAB=E:\IsaacLab\isaaclab.bat

REM  Rigid body prim NAMES to keep collision ON (chassis + all wheels)
REM  Separate with commas, no spaces around commas
set KEEP_BODIES=tn__base_link1_wJ,tn__Wheel11_i7t6,tn__Wheel21_i7t6,tn__Wheel31_i7t6,tn__Wheel41_i7t6,tn__Wheel51_i7t6,tn__Wheel61_i7t6

REM  Joint prim NAMES to skip when locking arm joints
REM  (these are your Python-actuated / wheel joints — leave their drives alone)
set SKIP_JOINTS=Left_Front_Wheel,Left_Center_Wheel,Left_Rear_Wheel,Right_Front_Wheel,Right_Center_Wheel,Right_Rear_Wheel

REM ----------------------------------------------------------------
REM  ARGUMENT CHECK
REM ----------------------------------------------------------------
set TOOLS_DIR=%~dp0

if "%~1"=="" (
    echo.
    echo  Usage: run_usd_tools.bat "C:\path\to\robot.usd"
    echo.
    exit /b 1
)
set USD=%~1

echo.
echo ================================================================
echo   USD Isaac Lab Tool Pipeline
echo   %USD%
echo ================================================================

REM ----------------------------------------------------------------
REM  STEP 1 — Compliance check (read-only)
REM ----------------------------------------------------------------
echo.
echo ---- STEP 1 / 3 :  Compliance Check  (read-only) ----
echo.
call "%ISAACLAB%" -p "%TOOLS_DIR%check_usd_compliance.py" --usd "%USD%" --headless
if errorlevel 1 (
    echo.
    echo [ERROR] Compliance check failed to run. Aborting.
    exit /b 1
)

REM ----------------------------------------------------------------
REM  STEP 2 — Disable non-wheel collision
REM ----------------------------------------------------------------
echo.
echo ---- STEP 2 / 3 :  Disable Non-Wheel Collision ----
echo.
call "%ISAACLAB%" -p "%TOOLS_DIR%disable_nonwheel_collision.py" ^
    --usd "%USD%" ^
    --keep "%KEEP_BODIES%" ^
    --headless
if errorlevel 1 (
    echo.
    echo [ERROR] disable_nonwheel_collision.py failed. Aborting.
    exit /b 1
)

REM ----------------------------------------------------------------
REM  STEP 3 — Lock arm joints to position-hold
REM ----------------------------------------------------------------
echo.
echo ---- STEP 3 / 4 :  Lock Arm Joints ----
echo.
call "%ISAACLAB%" -p "%TOOLS_DIR%lock_arm_joints.py" ^
    --usd "%USD%" ^
    --skip "%SKIP_JOINTS%" ^
    --headless
if errorlevel 1 (
    echo.
    echo [ERROR] lock_arm_joints.py failed.
    exit /b 1
)

REM ----------------------------------------------------------------
REM  STEP 4 — Deactivate OmniGraph nodes (ROS2 controllers etc.)
REM ----------------------------------------------------------------
echo.
echo ---- STEP 4 / 4 :  Deactivate OmniGraph Nodes ----
echo.
call "%ISAACLAB%" -p "%TOOLS_DIR%deactivate_omnigraphs.py" ^
    --usd "%USD%" ^
    --headless
if errorlevel 1 (
    echo.
    echo [ERROR] deactivate_omnigraphs.py failed.
    exit /b 1
)

echo.
echo ================================================================
echo   Done. Run your compliance check again to verify:
echo   run_usd_tools.bat "%USD%"
echo   (or just re-run step 1 manually)
echo ================================================================
echo.
endlocal
