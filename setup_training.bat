@echo off
echo Installing PyTorch with CUDA 12.6 support...
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
echo.
echo Installing remaining dependencies...
pip install -r training/requirements.txt
echo.
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND — training will be slow on CPU')"
echo.
echo Done. To start training:
echo   python training/train.py
echo.
echo To train with live loss plot:
echo   python training/train.py --view
pause
