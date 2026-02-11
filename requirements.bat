@echo off
echo [1/3] Installazione Manuale Torch (GPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/3] Installazione Librerie Comuni...
pip install -r requirements.txt

echo.
echo [3/3] Installazione Manuale TTS (No-Deps)...
pip install TTS==0.22.0 --no-deps

echo.
echo PIPELINE COMPLETATA.
pause