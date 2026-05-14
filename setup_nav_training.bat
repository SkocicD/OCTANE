@echo off
setlocal

echo.
echo  OCTANE Navigation Model -- Environment Setup
echo  =============================================
echo.

:: ── Python version check (3.10+ required for | union type syntax) ────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 or newer from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause & exit /b 1
)

python -c "import sys; code=0 if sys.version_info>=(3,10) else 1; print(f'Python {sys.version.split()[0]}'); exit(code)" 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    echo         This code uses X ^| None union syntax ^(PEP 604^) which does not exist in older versions.
    echo         Download from https://python.org
    pause & exit /b 1
)
echo [OK] Python version

:: ── NVIDIA driver check ──────────────────────────────────────────────────────
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN] nvidia-smi not found -- either no NVIDIA GPU or drivers not installed.
    echo        Training will run on CPU ^(very slow^).
    echo        For GPU: install NVIDIA drivers ^>=527 from https://nvidia.com/drivers
    echo        CUDA 12.6 support requires driver version 527 or newer on Windows.
    echo.
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name,driver_version --format^=csv,noheader 2^>nul') do (
        echo [OK] GPU: %%i
    )
)

:: ── Install PyTorch (CUDA 12.6) ──────────────────────────────────────────────
echo.
echo Installing PyTorch with CUDA 12.6 support...
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause & exit /b 1
)

:: ── Install remaining dependencies ───────────────────────────────────────────
echo.
echo Installing remaining dependencies...
pip install -r training_nav/requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies from training_nav/requirements.txt
    pause & exit /b 1
)

:: ── Final GPU sanity check via torch ─────────────────────────────────────────
echo.
python -c "import torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None; print('[OK] torch GPU:', gpu) if gpu else print('[WARN] torch: no CUDA GPU found -- training will use CPU')"

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo  Setup complete. Quick reference:
echo.
echo    Train (with prompts):      python training_nav/train.py
echo    Train (headless):          python training_nav/train.py --headless
echo    Train + live dashboard:    python training_nav/train.py --view
echo    Visualize arenas/model:    visualize_nav.bat
echo                               python training_nav/visualize.py --seed 42
echo.
echo  Prerequisites for a new machine:
echo    1. Python 3.10+      https://python.org
echo    2. Git               https://git-scm.com
echo    3. NVIDIA driver 527+ https://nvidia.com/drivers  (for GPU training)
echo.
pause
