@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating virtual environment and installing dependencies...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment. Is Python 3.11 or newer installed and on PATH?
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created .env from .env.example. Edit it to add your NVIDIA NIM key if you use NIM.
)

echo Starting Resume to JSON at http://localhost:8501  (Ctrl+C to stop)
rem Open the browser a few seconds from now, once the server is up.
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true

if errorlevel 1 pause
