"""
MedPredict AI — Model Dashboard Page
Shows model performance, comparison charts, and dataset statistics.
"""
import os
import streamlit as st
import json
import pandas as pd
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

st.set_page_config(page_title="Model Dashboard | MedPredict AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-highlight {
        background: linear-gradient(135deg, #1a73e8, #1557b0);
        border-radius: 12px; padding: 1.25rem; color: white; text-align: center;
    }
    .metric-highlight .val { font-size: 2rem; font-weight: 700; }
    .metric-highlight .lbl { font-size: 0.8rem; opacity: 0.85; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Model Dashboard")
st.caption("Performance metrics, model comparison, and dataset statistics for MedPredict AI")
st.markdown("---")

# ── Load Stats ──────────────────────────────────────────────────────────────────
stats_path = Path(__file__).parent.parent.parent / "models" / "model_stats.json"
dataset_path = Path(__file__).parent.parent.parent / "dataset" / "diseases.csv"

if not stats_path.exists():
    st.error("Model stats not found. Run `python notebooks/train_model.py` first.")
    st.stop()

with open(stats_path) as f:
    stats = json.load(f)

best_name = stats.get("best_model", "RandomForest")

def normalize_metrics(model_metrics):
    return {
        **model_metrics,
        "precision": model_metrics.get("precision_weighted", model_metrics.get("precision", 0)),
        "recall": model_metrics.get("recall_weighted", model_metrics.get("recall", 0)),
        "f1": model_metrics.get("f1_weighted", model_metrics.get("f1", 0)),
    }


rf_stats = normalize_metrics(stats.get("RandomForest", {}))
xgb_stats = normalize_metrics(stats.get("XGBoost", {}))

# ── Top Metrics Row ─────────────────────────────────────────────────────────────
st.markdown("### 🏆 Best Model Performance")
best = rf_stats if best_name == "RandomForest" else xgb_stats

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f'<div class="metric-highlight"><div class="val">{best_name.replace("RandomForest","RF")}</div><div class="lbl">Active Model</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-highlight"><div class="val">{best["accuracy"]}%</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-highlight"><div class="val">{best["precision"]}%</div><div class="lbl">Precision</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-highlight"><div class="val">{best["recall"]}%</div><div class="lbl">Recall</div></div>', unsafe_allow_html=True)
c5.markdown(f'<div class="metric-highlight"><div class="val">{best["f1"]}%</div><div class="lbl">F1 Score</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Side-by-side Comparison ─────────────────────────────────────────────────────
st.markdown("### ⚔️ Model Comparison")
col_l, col_r = st.columns(2)

def model_card(name, s, is_best):
    badge = " 🏆 **Best**" if is_best else ""
    st.markdown(f"#### {name}{badge}")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy",  f"{s['accuracy']}%")
    m2.metric("Precision", f"{s['precision']}%")
    m3, m4 = st.columns(2)
    m3.metric("Recall",    f"{s['recall']}%")
    m4.metric("F1 Score",  f"{s['f1']}%")

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    values  = [s["accuracy"], s["precision"], s["recall"], s["f1"]]
    for metric, val in zip(metrics, values):
        st.markdown(f"**{metric}**")
        st.progress(val / 100)

with col_l:
    with st.container():
        model_card("🌲 Random Forest", rf_stats, best_name == "RandomForest")

with col_r:
    with st.container():
        model_card("⚡ XGBoost", xgb_stats, best_name == "XGBoost")

st.markdown("---")

# ── Experiment Leaderboard ─────────────────────────────────────────────────────
st.markdown("### 🧪 Training Experiment Leaderboard")
api_base_url = os.environ.get("API_URL", "http://localhost:8000")
try:
    experiments_response = requests.get(f"{api_base_url}/experiments", timeout=15)
    experiments_response.raise_for_status()
    experiment_data = experiments_response.json()
    best_experiment = experiment_data.get("best")
    if best_experiment:
        best_metrics = best_experiment.get("metrics", {})
        st.caption(
            f"Best by weighted F1: {best_experiment['name']} "
            f"({best_metrics.get('f1_weighted', 'N/A')}%)"
        )
    runs = experiment_data.get("runs", [])
    if runs:
        st.dataframe(
            [
                {
                    "Experiment": run["name"],
                    "Accuracy": run.get("metrics", {}).get("accuracy"),
                    "Weighted F1": run.get("metrics", {}).get("f1_weighted"),
                    "Timestamp": run.get("timestamp"),
                }
                for run in runs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No training experiments logged yet. Run the training pipeline to populate this table.")
except requests.RequestException:
    st.warning("Experiment leaderboard unavailable. Check that the FastAPI backend is running.")

st.markdown("---")

# ── Dataset Statistics ──────────────────────────────────────────────────────────
st.markdown("### 📂 Dataset Statistics")

if dataset_path.exists():
    df = pd.read_csv(dataset_path)
    symptom_cols = [c for c in df.columns if c != "disease"]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Samples",   f"{len(df):,}")
    d2.metric("Diseases",        df["disease"].nunique())
    d3.metric("Symptoms",        len(symptom_cols))
    d4.metric("Train/Test Split", "80 / 20")

    st.markdown("#### Disease Distribution")
    disease_counts = df["disease"].value_counts().reset_index()
    disease_counts.columns = ["Disease", "Count"]
    st.bar_chart(disease_counts.set_index("Disease"))

    st.markdown("#### Top 20 Most Frequent Symptoms")
    sym_freq = df[symptom_cols].sum().sort_values(ascending=False).head(20).reset_index()
    sym_freq.columns = ["Symptom", "Frequency"]
    sym_freq["Symptom"] = sym_freq["Symptom"].str.replace("_", " ").str.title()
    st.bar_chart(sym_freq.set_index("Symptom"))

    st.markdown("#### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.warning("Dataset file not found.")

st.markdown("---")

# ── Training Config ──────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Training Configuration")
config_data = {
    "Parameter": [
        "RF: n_estimators", "RF: max_depth", "RF: min_samples_split",
        "XGB: n_estimators", "XGB: max_depth", "XGB: learning_rate",
        "Test size", "Random seed", "Cross-validation"
    ],
    "Value": [
        "200", "15", "4",
        "200", "6", "0.1",
        "20%", "42", "Stratified"
    ]
}
st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

# ── EDA Images ───────────────────────────────────────────────────────────────────
eda_path = Path(__file__).parent.parent.parent / "dataset" / "eda_plots.png"
cm_path  = Path(__file__).parent.parent.parent / "dataset" / "confusion_matrix.png"
shap_path= Path(__file__).parent.parent.parent / "dataset" / "shap_importance.png"

if eda_path.exists() or cm_path.exists() or shap_path.exists():
    st.markdown("---")
    st.markdown("### 📈 Training Plots")
    st.caption("Run `notebooks/training.ipynb` to generate these plots.")

    if eda_path.exists():
        st.image(str(eda_path), caption="EDA: Disease & Symptom Distribution", use_container_width=True)
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Feature Importance", use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.85rem;'>"
    "MedPredict AI v2 — Model Dashboard • For educational purposes only"
    "</div>",
    unsafe_allow_html=True
)
