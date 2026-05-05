@echo off
echo Installing terrain model training dependencies...
pip install -r training/requirements.txt
echo.
echo Done. To start training:
echo   python training/train.py
echo.
echo To train with live loss plot:
echo   python training/train.py --view
pause
