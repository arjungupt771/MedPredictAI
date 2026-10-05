"""
MedPredict AI — About Page
Project overview, tech stack, API docs, and resume bullet.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

st.set_page_config(page_title="About | MedPredict AI", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .tech-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .resume-box {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        font-style: italic;
        color: #343a40;
        margin: 1rem 0;
    }
    .api-endpoint {
        background: #1e1e1e;
        color: #a8ff78;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# ℹ️ About MedPredict AI")
st.caption("End-to-end AI-powered disease prediction platform")
st.markdown("---")

# ── Project Overview ────────────────────────────────────────────────────────────
st.markdown("## 🎯 What Is This?")
st.markdown("""
**MedPredict AI v2** is a full-stack machine learning project that predicts diseases from symptoms.

Unlike typical student projects that stop at `Dataset → Model → Accuracy`, this is built like an **actual product**:
""")

st.markdown("""
```
Dataset (2,400 samples, 20 diseases, 57 symptoms)
   ↓
ML Models (Random Forest + XGBoost, 94%+ accuracy)
   ↓
SHAP Explainability (why did the AI predict this?)
   ↓
FastAPI Backend (REST API with Swagger docs)
   ↓
Streamlit Frontend (multi-page app with charts)
   ↓
PDF Report Generation (downloadable patient reports)
   ↓
Docker Deployment (containerized, Render-ready)
```
""")

st.markdown("---")

# ── Tech Stack ──────────────────────────────────────────────────────────────────
st.markdown("## 🛠️ Tech Stack")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Machine Learning**")
    for tech in ["Scikit-Learn", "XGBoost", "Pandas", "NumPy", "Joblib"]:
        st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

with col2:
    st.markdown("**Backend & Explainability**")
    for tech in ["FastAPI", "Uvicorn", "Pydantic", "SHAP", "ReportLab"]:
        st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

with col3:
    st.markdown("**Frontend & DevOps**")
    for tech in ["Streamlit", "Docker", "Docker Compose", "Render", "Jupyter"]:
        st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

st.markdown("---")

# ── API Reference ───────────────────────────────────────────────────────────────
st.markdown("## 🌐 API Reference")
st.caption("FastAPI backend runs at `http://localhost:8000` — interactive docs at `/docs`")

endpoints = [
    ("GET",  "/",           "API info and available endpoints"),
    ("GET",  "/symptoms",   "List all 57 supported symptoms"),
    ("POST", "/predict",    "Predict disease from symptoms (top-3 + confidence)"),
    ("POST", "/explain",    "Get SHAP explainability for a prediction"),
    ("POST", "/report",     "Generate and download PDF report"),
    ("GET",  "/model-stats","Model comparison stats (RF vs XGBoost)"),
]

cols = st.columns([1, 2, 4])
cols[0].markdown("**Method**"); cols[1].markdown("**Endpoint**"); cols[2].markdown("**Description**")
st.markdown("---")
for method, path, desc in endpoints:
    c1, c2, c3 = st.columns([1, 2, 4])
    color = "#34a853" if method == "GET" else "#1a73e8"
    c1.markdown(f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:700;">{method}</span>', unsafe_allow_html=True)
    c2.markdown(f"`{path}`")
    c3.markdown(desc)

st.markdown("---")

# ── Example Request ──────────────────────────────────────────────────────────────
st.markdown("## 🔌 Example API Usage")
col_req, col_res = st.columns(2)

with col_req:
    st.markdown("**Request — POST /predict**")
    st.code("""\
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "symptoms": ["fever", "cough", "headache"],
    "patient_name": "John Doe"
  }'
""", language="bash")

with col_res:
    st.markdown("**Response**")
    st.code("""\
{
  "top_prediction": {
    "disease": "Influenza",
    "confidence": 0.91,
    "confidence_pct": 91.0,
    "doctor": "General Physician"
  },
  "top3": [
    {"disease": "Influenza",  "confidence_pct": 91.0},
    {"disease": "COVID-19",   "confidence_pct": 84.0},
    {"disease": "Pneumonia",  "confidence_pct": 71.0}
  ],
  "low_confidence": false,
  "matched_symptoms": ["fever", "cough", "headache"],
  "unrecognized_symptoms": []
}
""", language="json")

st.markdown("---")

# ── Project Structure ────────────────────────────────────────────────────────────
st.markdown("## 📂 Project Structure")
st.code("""\
medpredict-ai/
├── backend/
│   ├── app.py          # FastAPI REST API
│   ├── predictor.py    # Core prediction engine
│   ├── explain.py      # SHAP explainability
│   ├── report.py       # PDF report generation
│   └── config.py       # Central configuration
├── frontend/
│   ├── app.py          # Main Streamlit UI
│   └── pages/
│       ├── 1_📊_Model_Dashboard.py
│       └── 2_ℹ️_About.py
├── models/
│   ├── model.pkl       # Best model
│   ├── rf_model.pkl    # Random Forest
│   ├── xgb_model.pkl   # XGBoost
│   ├── label_encoder.pkl
│   ├── symptoms.pkl
│   └── model_stats.json
├── dataset/
│   ├── diseases.csv
│   └── generate_dataset.py
├── notebooks/
│   └── training.ipynb  # Full training pipeline
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── requirements.txt
└── README.md
""", language="text")

st.markdown("---")

# ── Resume Bullet ────────────────────────────────────────────────────────────────
st.markdown("## 💼 Resume Bullet Point")
st.markdown("""
Copy this for your CV/LinkedIn:
""")
st.markdown("""
<div class="resume-box">
Built <strong>MedPredict AI</strong>, an end-to-end healthcare prediction platform using 
<strong>XGBoost &amp; Random Forest</strong> achieving <strong>94%+ accuracy</strong> across 
20 diseases and 57 symptoms. Implemented <strong>SHAP explainability</strong>, 
<strong>PDF report generation</strong>, <strong>FastAPI REST API</strong>, 
<strong>multi-page Streamlit frontend</strong>, and <strong>Dockerized deployment</strong>. 
Trained on 2,400 samples with model comparison dashboard and confidence-threshold-based 
fallback logic.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Quick Start ───────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Quick Start")
st.code("""\
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset + train models
python dataset/generate_dataset.py
python notebooks/train_model.py

# 3. Run backend (Terminal 1)
cd backend && uvicorn app:app --reload
# → http://localhost:8000/docs

# 4. Run frontend (Terminal 2)
cd frontend && streamlit run app.py
# → http://localhost:8501

# 5. Or run everything with Docker
cd docker && docker-compose up --build
""", language="bash")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.85rem;'>"
    "MedPredict AI v2 • Educational Project • Not for clinical use"
    "</div>",
    unsafe_allow_html=True
)
