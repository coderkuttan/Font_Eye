@echo off
rem FontEye launcher - works from any folder. Use FontEye.bat --snip to snip immediately.
cd /d "%~dp0"
start "" "%~dp0.venv\Scripts\pythonw.exe" -m fonteye %*
