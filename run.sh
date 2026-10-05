#!/bin/bash
# MedPredict AI — Quick Start Script
# Usage: ./run.sh [setup|backend|frontend|all|docker]

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_banner() {
    echo -e "${BLUE}"
    echo "  ╔══════════════════════════════════╗"
    echo "  ║       🏥  MedPredict AI v2       ║"
    echo "  ╚══════════════════════════════════╝"
    echo -e "${NC}"
}

setup() {
    echo -e "${YELLOW}[1/3] Installing dependencies...${NC}"
    pip install -r requirements.txt -q

    echo -e "${YELLOW}[2/3] Generating dataset...${NC}"
    python dataset/generate_dataset.py

    echo -e "${YELLOW}[3/3] Training models...${NC}"
    python notebooks/train_model.py

    echo -e "${GREEN}✅ Setup complete! Models saved to models/${NC}"
}

run_backend() {
    echo -e "${GREEN}▶ Starting FastAPI backend on http://localhost:8000${NC}"
    echo -e "  API docs: http://localhost:8000/docs"
    cd backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000
}

run_frontend() {
    echo -e "${GREEN}▶ Starting Streamlit frontend on http://localhost:8501${NC}"
    cd frontend && streamlit run app.py --server.port 8501
}

run_all() {
    echo -e "${GREEN}Starting backend + frontend...${NC}"
    # Start backend in background
    (cd backend && uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1) &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID (logs: /tmp/backend.log)"

    sleep 2

    # Start frontend (foreground)
    cd frontend && streamlit run app.py --server.port 8501

    # Cleanup on exit
    kill $BACKEND_PID 2>/dev/null
}

run_docker() {
    echo -e "${GREEN}▶ Starting with Docker Compose...${NC}"
    cd docker && docker-compose up --build
}

print_banner

case "${1:-all}" in
    setup)    setup ;;
    backend)  run_backend ;;
    frontend) run_frontend ;;
    all)      run_all ;;
    docker)   run_docker ;;
    *)
        echo "Usage: ./run.sh [setup|backend|frontend|all|docker]"
        echo ""
        echo "  setup    — Install deps + generate data + train models"
        echo "  backend  — Start FastAPI only"
        echo "  frontend — Start Streamlit only"
        echo "  all      — Start both (default)"
        echo "  docker   — Start via docker-compose"
        ;;
esac
