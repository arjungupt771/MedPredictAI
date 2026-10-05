"""
MedPredict AI — Configuration
Central config for paths, thresholds, and settings.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "dataset"

MODEL_PATH    = MODEL_DIR / "model.pkl"
RF_MODEL_PATH = MODEL_DIR / "rf_model.pkl"
XGB_MODEL_PATH= MODEL_DIR / "xgb_model.pkl"
ENCODER_PATH  = MODEL_DIR / "label_encoder.pkl"
SYMPTOMS_PATH = MODEL_DIR / "symptoms.pkl"
STATS_PATH    = MODEL_DIR / "model_stats.json"

# ── Prediction Settings ────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.60   # Below this → warn user to see a doctor
TOP_N_PREDICTIONS        = 3      # Number of differential diagnoses to return

# ── API Settings ───────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Frontend Settings ──────────────────────────────────────────────────────────
STREAMLIT_PORT = 8501
API_BASE_URL   = "http://localhost:8000"  # Override via env var API_URL

# ── SHAP Settings ──────────────────────────────────────────────────────────────
SHAP_MAX_DISPLAY = 8    # Number of top factors to show in explanation

# ── Doctor Recommendations ─────────────────────────────────────────────────────
DOCTOR_MAP = {
    "Influenza":       "General Physician",
    "Common Cold":     "General Physician",
    "COVID-19":        "Infectious Disease Specialist",
    "Pneumonia":       "Pulmonologist",
    "Bronchitis":      "Pulmonologist",
    "Malaria":         "Infectious Disease Specialist",
    "Dengue":          "Infectious Disease Specialist",
    "Typhoid":         "Infectious Disease Specialist",
    "Jaundice":        "Gastroenterologist / Hepatologist",
    "Diabetes":        "Endocrinologist",
    "Hypertension":    "Cardiologist",
    "Gastroenteritis": "Gastroenterologist",
    "Migraine":        "Neurologist",
    "Asthma":          "Pulmonologist / Allergist",
    "Appendicitis":    "General Surgeon (Emergency)",
    "UTI":             "Urologist / General Physician",
    "Chickenpox":      "Dermatologist / General Physician",
    "Measles":         "General Physician / Pediatrician",
    "Tuberculosis":    "Pulmonologist",
    "Anemia":          "Hematologist",
}
