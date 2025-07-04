@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM Exit on any error
SETLOCAL ENABLEDELAYEDEXPANSION
REM Trap errors manually: after each critical command we check ERRORLEVEL
REM ─────────────────────────────────────────────────────────────────────────────

REM Function: error_exit [message]
:ERROR_EXIT
  echo %~1 1>&2
  echo %DATE% %TIME% - %~1 >> error.log
  ENDLOCAL
  exit /b 1

REM 1. ENV_NAME default
if not defined ENV_NAME (
  set "ENV_NAME=phishpedia"
)

REM 2. (always set now)

REM 3. Retry count
set "RETRY_COUNT=3"

REM 4. download_with_retry [file_id] [file_name]
:DOWNLOAD_WITH_RETRY
  set "FILE_ID=%~1"
  set "FILE_NAME=%~2"
  set /A "COUNT=0"
  :DL_LOOP
    set /A "ATTEMPT=COUNT + 1"
    echo Attempting to download !FILE_NAME! (Attempt !ATTEMPT!/!RETRY_COUNT!)...
    conda run -n "%ENV_NAME%" gdown --id "!FILE_ID!" -O "!FILE_NAME!"
    if not errorlevel 1 (
      goto DL_DONE
    )
    set /A "COUNT+=1"
    if !COUNT! geq !RETRY_COUNT! (
      call :ERROR_EXIT "Failed to download !FILE_NAME! after !RETRY_COUNT! attempts."
    )
    echo Retry !COUNT! of !RETRY_COUNT! for !FILE_NAME!...
    timeout /t 2 /nobreak >nul
    goto DL_LOOP
  :DL_DONE
  exit /b 0

REM 5. Ensure conda is installed
where conda >nul 2>&1
if errorlevel 1 (
  call :ERROR_EXIT "Conda is not installed. Please install Anaconda/Miniconda and try again."
)

REM 6. Initialize conda for cmd.exe (assumes conda initialized in cmd)
REM If you have problems, uncomment the next two lines and adjust path:
REM set "CONDA_BASE=C:\Users\%USERNAME%\Anaconda3"
REM call "%CONDA_BASE%\condabin\conda-hook.bat"

REM 7. Check if environment exists
conda env list | findstr /R /C:"^%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
  echo Creating new Conda environment: %ENV_NAME% with Python 3.10...
  conda create -y -n "%ENV_NAME%" python=3.10 || call :ERROR_EXIT "Failed to create environment %ENV_NAME%."
) else (
  echo Activating existing Conda environment: %ENV_NAME%...
)

REM 8. Activate
call conda activate "%ENV_NAME%" || call :ERROR_EXIT "Failed to activate environment %ENV_NAME%."

REM 9. Install dependencies
echo Installing PyTorch & friends...
conda install -y -n "%ENV_NAME%" pytorch torchvision torchaudio cpuonly -c pytorch || call :ERROR_EXIT "Failed to install PyTorch."
echo Installing pip packages...
pip install gdown opencv-python numpy Pillow matplotlib || call :ERROR_EXIT "Failed to pip install dependencies."

REM 10. Prepare models directory
if not exist models mkdir models
pushd models

REM 11. Download model checkpoint
call :DOWNLOAD_WITH_RETRY "1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS" "resnetv2_rgb_new.pth.tar"

REM 12. Download and unzip expand_targetlist
call :DOWNLOAD_WITH_RETRY "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" "expand_targetlist.zip"

REM Use PowerShell's Expand-Archive for zip extraction
powershell -Command "Expand-Archive -LiteralPath 'expand_targetlist.zip' -DestinationPath 'expand_targetlist' -Force" \
  || call :ERROR_EXIT "Failed to unzip expand_targetlist.zip."

pushd expand_targetlist 2>nul || call :ERROR_EXIT "Directory expand_targetlist not found after unzip."

REM 13. Handle nested expand_targetlist\expand_targetlist
if exist expand_targetlist (
  echo Nested directory 'expand_targetlist\' detected. Moving contents up...
  pushd expand_targetlist
    for /F "delims=" %%F in ('dir /A:-D /B /O:N') do move "%%F" ".."
  popd
  rd /S /Q expand_targetlist
)
popd  REM back to models
popd  REM back to original

echo.
echo Extraction and installation completed successfully!
echo.

ENDLOCAL
exit /b 0
