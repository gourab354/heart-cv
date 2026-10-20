# Heart Vision PowerShell Launcher
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Starting Heart Vision: Real-Time Webcam rPPG Heart Rate Detector" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

python main.py $args

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[!] An error occurred while launching Heart Vision." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit..."
}
