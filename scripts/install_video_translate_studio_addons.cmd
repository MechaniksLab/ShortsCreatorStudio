@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "RUNTIME_PYTHON=%SCRIPT_DIR%..\runtime\python.exe"

if not exist "%RUNTIME_PYTHON%" (
  echo [ERROR] Runtime python not found: %RUNTIME_PYTHON%
  exit /b 2
)

"%RUNTIME_PYTHON%" "%SCRIPT_DIR%setup_video_translate_runtime.py" --runtime-python "%RUNTIME_PYTHON%" --profile none --upgrade-pip --with-studio-addons
if errorlevel 1 exit /b %errorlevel%

"%RUNTIME_PYTHON%" "%SCRIPT_DIR%check_video_translate_env.py" --runtime-python "%RUNTIME_PYTHON%"
exit /b %errorlevel%
