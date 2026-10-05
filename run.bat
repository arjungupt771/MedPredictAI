@echo off
REM MedPredict AI — Quick Start Script (Windows)
REM Usage: run.bat [setup|backend|frontend|all]

echo.
echo   ╔══════════════════════════════════╗
echo   ║       MedPredict AI v2           ║
echo   ╚══════════════════════════════════╝
echo.

if "%1"=="setup" goto setup
if "%1"=="backend" goto backend
if "%1"=="frontend" goto frontend
if "%1"=="all" goto all
goto all

:setup
echo [1/3] Installing dependencies...
pip install -r requirements.txt -q
echo [2/3] Generating dataset...
python dataset\generate_dataset.py
echo [3/3] Training models...
python notebooks\train_model.py
echo Done! Models saved to models\
goto end

:backend
echo Starting FastAPI backend on http://localhost:8000
cd backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000
goto end

:frontend
echo Starting Streamlit frontend on http://localhost:8501
cd frontend && streamlit run app.py --server.port 8501
goto end

:all
echo Starting backend in background...
start "MedPredict Backend" cmd /k "cd backend && uvicorn app:app --reload --port 8000"
timeout /t 3 /nobreak > NUL
echo Starting frontend...
cd frontend && streamlit run app.py --server.port 8501
goto end

:end
