@echo off
rem ---------------------------------------------------------------
rem  Glassbox launcher (Windows)
rem    run.bat                 start the UI
rem    run.bat --rebuild       re-index corpus/ then start
rem    run.bat --with-llm      also install llama-cpp-python + the GGUF
rem ---------------------------------------------------------------
setlocal enabledelayedexpansion
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe

if not exist "%PY%" (
  echo [glassbox] creating virtualenv .venv ...
  python -m venv "%HERE%.venv" || goto :err
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r "%HERE%requirements.txt" || goto :err
)

"%PY%" -c "import glassbox" 2>nul
if errorlevel 1 (
  echo [glassbox] installing package in editable mode ...
  "%PY%" -m pip install -e "%HERE%." || goto :err
)

rem Idempotent: seeds the demo corpus only when corpus\ holds no documents, so a
rem fresh clone (where .gitkeep keeps the folder but leaves it empty) still gets
rem the 12 demo documents while your own notes are never overwritten.
"%PY%" "%HERE%scripts\seed_corpus.py"

set WITHLLM=0
for %%a in (%*) do if "%%a"=="--with-llm" set WITHLLM=1
if "%WITHLLM%"=="1" (
  "%PY%" -c "import llama_cpp" 2>nul || (
    echo [glassbox] installing llama-cpp-python prebuilt wheel ...
    "%PY%" -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama-cpp-python
  )
  "%PY%" -m glassbox download-model
)

set REBUILD=
for %%a in (%*) do if "%%a"=="--rebuild" set REBUILD=--rebuild

echo.
echo [glassbox] http://127.0.0.1:8765
start "" http://127.0.0.1:8765
"%PY%" -m glassbox serve --port 8765 %REBUILD%
goto :eof

:err
echo.
echo [glassbox] setup failed - see the messages above.
exit /b 1
