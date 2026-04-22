@echo off
setlocal
set "EXIT_CODE=0"

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "RUNTIME_DIR=%PROJECT_DIR%\runtime_rvc"
set "RUNTIME_PYTHON=%RUNTIME_DIR%\Scripts\python.exe"

if not exist "%RUNTIME_PYTHON%" (
  echo [INFO] Creating runtime_rvc with Python 3.10...
  py -3.10 -m venv "%RUNTIME_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv with Python 3.10. Install Python 3.10 and retry.
    set "EXIT_CODE=2"
    goto :END
  )
)

echo [INFO] Upgrade pip/setuptools/wheel...
"%RUNTIME_PYTHON%" -m pip install --index-url https://pypi.org/simple --upgrade "pip<24.1" "setuptools<82" wheel
if errorlevel 1 goto :FAIL

echo [INFO] Clear pip cache...
"%RUNTIME_PYTHON%" -m pip cache purge

echo [INFO] Installing torch/torchaudio (CUDA 12.4)...
"%RUNTIME_PYTHON%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
if errorlevel 1 goto :FAIL

echo [INFO] Installing native RVC package...
set "READTHEDOCS=1"
"%RUNTIME_PYTHON%" -m pip install --no-cache-dir --index-url https://pypi.org/simple rvc-python
if errorlevel 1 goto :FAIL

echo [INFO] Smoke check...
"%RUNTIME_PYTHON%" -c "import importlib,sys; ok=False; errs=[]; [errs.append(n) for n in ('rvc_python','rvc') if not ok and not (__import__('builtins'), False)];\
for n in ('rvc_python','rvc'):\
  try:\
    importlib.import_module(n); print('OK', n); ok=True; break\
  except Exception as e:\
    pass;\
sys.exit(0 if ok else 1)"
if errorlevel 1 goto :FAIL

echo.
echo [OK] RVC runtime is ready: %RUNTIME_PYTHON%
goto :END

:FAIL
set "EXIT_CODE=%errorlevel%"

:END
echo.
if not "%EXIT_CODE%"=="0" echo [ERROR] Installer finished with code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
