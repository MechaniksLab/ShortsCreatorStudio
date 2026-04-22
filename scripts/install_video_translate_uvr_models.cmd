@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "RUNTIME_PYTHON=%SCRIPT_DIR%..\runtime\python.exe"
set "DOWNLOADER=%SCRIPT_DIR%download_uvr_models.py"
set "CHECKER=%SCRIPT_DIR%check_video_translate_env.py"

title UVR Models Installer (Video Translate)
echo ================================================
echo   Video Translate - UVR Models Installer
echo ================================================
echo.

if not exist "%RUNTIME_PYTHON%" (
  echo [ERROR] Runtime python not found:
  echo         %RUNTIME_PYTHON%
  echo.
  pause
  exit /b 2
)

if not exist "%DOWNLOADER%" (
  echo [ERROR] Downloader script not found:
  echo         %DOWNLOADER%
  echo.
  pause
  exit /b 3
)

echo [RUN] Downloading UVR models (Inst_HQ_3 + Kim_Vocal_2)...
"%RUNTIME_PYTHON%" "%DOWNLOADER%"
set "DL_RC=%ERRORLEVEL%"

echo.
echo [RUN] Environment check...
"%RUNTIME_PYTHON%" "%CHECKER%" --runtime-python "%RUNTIME_PYTHON%"
set "CHK_RC=%ERRORLEVEL%"

echo.
if not "%DL_RC%"=="0" (
  echo [WARN] UVR models download finished with code %DL_RC%.
  echo        If Kim_Vocal_2 was not downloaded, likely network/provider block.
  echo        Try again later or via VPN/proxy.
  echo.
) else (
  echo [OK] UVR models downloaded successfully.
)

if not "%CHK_RC%"=="0" (
  echo [WARN] Env check finished with code %CHK_RC%.
)

echo ================================================
echo Done.
echo ================================================
pause
exit /b 0
