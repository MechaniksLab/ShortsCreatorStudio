@echo off
setlocal
set "EXIT_CODE=0"

set "SCRIPT_DIR=%~dp0"
set "RUNTIME_PYTHON=%SCRIPT_DIR%..\runtime\python.exe"

if not exist "%RUNTIME_PYTHON%" (
  echo [ERROR] Runtime python not found: %RUNTIME_PYTHON%
  set "EXIT_CODE=2"
  goto :END
)

"%RUNTIME_PYTHON%" "%SCRIPT_DIR%setup_video_translate_runtime.py" --runtime-python "%RUNTIME_PYTHON%" --profile none --upgrade-pip --with-uvr-models --with-rvc-native
if errorlevel 1 goto :FAIL

"%RUNTIME_PYTHON%" "%SCRIPT_DIR%check_video_translate_env.py" --runtime-python "%RUNTIME_PYTHON%"
if errorlevel 1 goto :FAIL

echo [OK] Done.
goto :END

:FAIL
set "EXIT_CODE=%errorlevel%"

:END
echo.
if not "%EXIT_CODE%"=="0" echo [ERROR] Installer finished with code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
