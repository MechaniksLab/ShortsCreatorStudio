@echo off
setlocal
set "EXIT_CODE=0"

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

set "GPU_NAMES="
for /f "delims=" %%G in ('powershell -NoProfile -Command "$g=(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '; Write-Output $g"') do set "GPU_NAMES=%%G"

set "HAS_NVIDIA=0"
if defined GPU_NAMES (
  echo [INFO] Detected GPU(s): %GPU_NAMES%
  if /I not "%GPU_NAMES:NVIDIA=%"=="%GPU_NAMES%" set "HAS_NVIDIA=1"
) else (
  echo [WARN] Could not detect GPU names, fallback strategy will be used.
)

echo.
echo ==================================================
echo   Video Translate Full Auto Installer
echo ==================================================
echo [INFO] Step 1/2: install main runtime dependencies

if "%HAS_NVIDIA%"=="1" (
  echo [INFO] NVIDIA GPU detected -> running gpu-auto profile
  call "%SCRIPT_DIR%install_video_translate_gpu_auto.cmd"
) else (
  echo [INFO] Non-NVIDIA GPU (or unknown) -> running stable CPU profile
  call "%SCRIPT_DIR%install_video_translate_cpu.cmd"
)
if errorlevel 1 goto :FAIL

echo.
echo [INFO] Step 2/2: install dedicated RVC runtime (Python 3.10, optional but recommended for native RVC)

if "%HAS_NVIDIA%"=="1" (
  echo [INFO] Trying RVC CUDA 12.8 first...
  call "%SCRIPT_DIR%install_video_translate_rvc_runtime_gpu_cu128.cmd"
  if errorlevel 1 (
    echo [WARN] RVC CUDA 12.8 install failed, trying CUDA 12.4...
    call "%SCRIPT_DIR%install_video_translate_rvc_runtime_gpu_cu124.cmd"
    if errorlevel 1 (
      echo [WARN] RVC dedicated runtime install failed for both CUDA 12.8 and 12.4.
      echo [WARN] Main video-translate runtime is still installed.
      echo [WARN] You can continue without native RVC, or retry RVC installer manually.
    )
  )
) else (
  echo [INFO] Dedicated RVC GPU runtime skipped for non-NVIDIA setup.
  echo [INFO] You can keep using video-translate without native RVC,
  echo [INFO] or configure a custom RVC runtime path later in Settings.
)

echo.
echo [OK] Full auto installer finished.
goto :END

:FAIL
set "EXIT_CODE=%errorlevel%"

:END
echo.
if not "%EXIT_CODE%"=="0" echo [ERROR] Installer finished with code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
