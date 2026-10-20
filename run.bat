@echo off
title Heart Vision - Real-Time Heart Rate Detector
echo ==============================================================
echo   Starting Heart Vision: Real-Time Webcam rPPG Heart Rate Detector
echo ==============================================================
echo.
python main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] An error occurred while launching Heart Vision.
    pause
)
