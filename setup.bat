@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo === OCR-Video first-time setup ===
echo Project: %cd%
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ and try again.
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment: venv
    python -m venv venv
    if errorlevel 1 exit /b 1
)

call "%~dp0venv\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Exporting LTX-Video to OpenVINO IR ^(large download; may take a long time^)...
python export_ltx.py
if errorlevel 1 (
    echo LTX export failed. Fix the error above, then run: python export_ltx.py
    exit /b 1
)

echo.
echo Prefetching TrOCR weights ^(microsoft/trocr-large-handwritten^)...
python -c "import yaml; import ocr_ov; cfg = yaml.safe_load(open('config.yaml', encoding='utf-8')); ocr_ov.init_from_config(cfg); print('TrOCR ready.')"
if errorlevel 1 (
    echo TrOCR prefetch failed. It will retry on first Recognize in the app.
)

echo.
echo Setup finished.
echo Start the app with: run.bat
echo.
exit /b 0
