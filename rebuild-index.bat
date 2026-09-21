@echo off
setlocal
echo ==== Checking the Python packages needed for the search index ====
python -c "import sentence_transformers, numpy" 2>nul
if errorlevel 1 (
  echo Installing the packages from requirements.txt - first time only, this can take a few minutes...
  python -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo.
    echo FAILED: could not install requirements.txt - see the messages above.
    pause
    exit /b 1
  )
)

echo.
echo ==== Rebuilding the search index ====
python build_embeddings.py
if errorlevel 1 (
  echo.
  echo FAILED: build_embeddings.py - see the messages above.
  pause
  exit /b 1
)

echo.
echo Done! In GitHub Desktop, commit and push everything - the data\index files and any new .md files.
pause
