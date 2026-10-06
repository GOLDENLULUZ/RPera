@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"

pushd "%~dp0"
if errorlevel 1 goto directory_failed

where uv >nul 2>&1
if errorlevel 1 goto missing_uv
call uv --version
if errorlevel 1 goto failed

echo Preparing Python and project dependencies. The first start may take a few minutes.
call uv sync --locked --no-dev
if errorlevel 1 goto failed

echo Press Ctrl+C in this window to stop RPera.
call uv run --no-sync python -c "import uvicorn; from rpera.server_config import load_server_config; config = load_server_config(); host = '0.0.0.0' if config.lan_access else '127.0.0.1'; print(f'Open http://127.0.0.1:{config.port} in your browser (listen: {host}).', flush=True); uvicorn.run('rpera.app:app', host=host, port=config.port)"
if errorlevel 1 goto failed

popd
exit /b 0

:missing_uv
echo.
echo uv was not found. Install it using this command in PowerShell:
echo powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
echo Then open a new terminal or double-click start.bat again.
goto failed

:failed
echo.
echo RPera could not start. Review the error above.
popd
pause
exit /b 1

:directory_failed
echo Could not open the project directory.
pause
exit /b 1
