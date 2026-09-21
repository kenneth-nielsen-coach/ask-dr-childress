@echo off
setlocal
set FAILED=0

echo ==== Installing or updating the tools these scripts need ====
python -m pip install --quiet --disable-pip-version-check -r requirements-update.txt
if errorlevel 1 (
  echo   WARNING: could not install requirements-update.txt
  set FAILED=1
)

call :run "Updating YouTube transcripts" childress_transcripts.py
call :run "Updating blog posts" childress_blog_scraper.py
call :run "Updating Substack posts" childress_substack_scraper.py
call :run "Updating papers posted" childress_papers.py
call :run "Updating pdf converters" childress_pdf_converter.py
call :run "Updating pages" convert_page_pdfs.py

rem The app searches the precomputed index in data\index, so it has to be
rem rebuilt after the content changes, otherwise new posts never show up.
python -c "import sentence_transformers" 2>nul
if errorlevel 1 (
  echo.
  echo ==== Skipping search index rebuild: sentence-transformers is not installed ====
  echo   Run once:  python -m pip install -r requirements.txt
  echo   Then run:  python build_embeddings.py
  set FAILED=1
) else (
  call :run "Rebuilding search index" build_embeddings.py
)

echo.
if "%FAILED%"=="0" (
  echo Done! Now commit and push in GitHub Desktop, including the data\index files.
) else (
  echo FINISHED WITH PROBLEMS - scroll up and look for FAILED or WARNING before you commit.
)
pause
goto :eof

:run
echo.
echo ==== %~1 ====
python %2
if errorlevel 1 (
  echo   FAILED: %2
  set FAILED=1
)
goto :eof
