@echo off
chcp 65001 > nul
echo ========================================================
echo   System One CPU Servisi Başlatılıyor...
echo   Adres: http://127.0.0.1:8150
echo ========================================================
cd /d "%~dp0"
python server.py
pause
