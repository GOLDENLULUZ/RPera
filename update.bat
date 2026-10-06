@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
rem Parse the whole block before updating this batch file itself.
(
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update.ps1"
    if errorlevel 1 (exit /b 1) else (exit /b 0)
)
