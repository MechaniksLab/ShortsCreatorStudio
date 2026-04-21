@echo off
setlocal

set RUNTIME_PYTHON=e:\Neyro\ShortsCreatorStudio\runtime\python.exe

if not exist "%RUNTIME_PYTHON%" (
  echo [ERROR] Runtime python not found: %RUNTIME_PYTHON%
  exit /b 2
)

"%RUNTIME_PYTHON%" "%~dp0setup_video_translate_runtime.py" --runtime-python "%RUNTIME_PYTHON%" --profile gpu-cu124 --upgrade-pip
if errorlevel 1 exit /b %errorlevel%

"%RUNTIME_PYTHON%" "%~dp0check_video_translate_env.py" --runtime-python "%RUNTIME_PYTHON%"
exit /b %errorlevel%
