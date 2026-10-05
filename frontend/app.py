"""
MedPredict AI - Streamlit Frontend (v2.1)
Talks to the FastAPI backend over HTTP — no more direct backend imports.
"""
import os
import streamlit as st
import requests

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")


def api_get(path, **kwargs):
    return requests.get(f"{API_BASE_URL}{path}", timeout=15, **kwargs)


def api_post(path, json_body):
    return requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=30)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        color: #6c757d;
        font-size: 1rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a73e8, #1557b0);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-top: 0.25rem; }
    
    .prediction-card {
        background: white;
        border: 2px solid #1a73e8;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .disease-name {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .confidence-label { color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem; }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .factor-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #1a73e8, #4285f4);
    }
    .disclaimer {
        background: #fff3f3;
        border: 1px solid #ea4335;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.8rem;
        color: #ea4335;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1a73e8, #34a853);
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    patient_name = st.text_input("Patient Name (optional)", placeholder="Enter name for PDF report")
    
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    
    try:
        stats = api_get("/model-stats").json()
        best = stats.get("best_model", "RandomForest")
        st.success(f"✅ Active Model: **{best}**")
        cv = stats.get("cross_validation")
        if cv:
            st.caption(f"5-fold CV accuracy: {cv['accuracy_mean']}% ± {cv['accuracy_std']}%")
        for model_name, data in stats.items():
            if isinstance(data, dict) and "accuracy" in data:
                st.markdown(f"**{model_name}**")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{data['accuracy']}%")
                col2.metric("Weighted F1", f"{data.get('f1_weighted', data.get('f1'))}%")
    except requests.RequestException:
        st.error("⚠️ Backend unreachable — is FastAPI running?")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **MedPredict AI v2** uses:
    - 🌲 Random Forest / XGBoost
    - 🔍 SHAP Explainability  
    - 📄 PDF Report Generation
    - 🚀 FastAPI + Streamlit
    """)
    
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ For educational purposes only. Not a medical device.</div>',
        unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🏥 MedPredict AI</div>
    <div class="main-subtitle">AI-Powered Disease Prediction with Explainable AI</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Symptom Selection ──────────────────────────────────────────────────────────
try:
    symptoms_resp = api_get("/symptoms").json()
    all_symptoms = symptoms_resp["symptoms"]
    display_symptoms = symptoms_resp["display"]
except requests.RequestException:
    st.error("⚠️ Could not load symptom list from the backend.")
    st.stop()

symptom_map = {d: r for d, r in zip(display_symptoms, all_symptoms)}

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🩺 Select Your Symptoms")
    st.caption("Choose all symptoms you are currently experiencing")
    
    selected_display = st.multiselect(
        "Symptoms",
        options=display_symptoms,
        placeholder="Start typing or scroll to find symptoms...",
        label_visibility="collapsed",
    )
    
    selected_symptoms = [symptom_map[d] for d in selected_display]
    
    if selected_symptoms:
        st.markdown(f"**{len(selected_symptoms)} symptom(s) selected:**")
        cols = st.columns(3)
        for i, s in enumerate(selected_display):
            cols[i % 3].markdown(f"✅ {s}")

with col_right:
    st.markdown("### 🔍 Quick Select")
    st.caption("Common symptom groups")
    
    preset_groups = {
        "🤒 Flu-like": ["fever", "cough", "headache", "body_aches", "fatigue"],
        "🌡️ Respiratory": ["cough", "shortness_of_breath", "chest_pain", "wheezing"],
        "🤢 GI Issues": ["nausea", "vomiting", "diarrhea", "stomach_cramps"],
        "😴 Weakness": ["fatigue", "weakness", "dizziness", "headache"],
    }
    
    for label, symptoms_list in preset_groups.items():
        if st.button(label, use_container_width=True):
            # Map to display format
            new_display = [s.replace("_", " ").title() for s in symptoms_list if s in all_symptoms]
            st.session_state["preset"] = new_display
            st.rerun()

# Apply preset if set
if "preset" in st.session_state and not selected_symptoms:
    selected_display = st.session_state.pop("preset", [])
    selected_symptoms = [symptom_map.get(d, d.lower().replace(" ", "_")) for d in selected_display]

st.markdown("---")

# ── Predict Button ─────────────────────────────────────────────────────────────
predict_col, _ = st.columns([2, 3])
with predict_col:
    predict_btn = st.button(
        "🔬 Predict Disease",
        type="primary",
        use_container_width=True,
        disabled=len(selected_symptoms) == 0
    )

if not selected_symptoms:
    st.info("👆 Select at least one symptom above to get a prediction")

# ── Results ────────────────────────────────────────────────────────────────────
if predict_btn and selected_symptoms:
    with st.spinner("🧠 Analyzing symptoms..."):
        resp = api_post("/predict", {"symptoms": selected_symptoms, "patient_name": patient_name or "Patient"})

    if resp.status_code == 422:
        detail = resp.json().get("detail", {})
        if detail.get("error"):
            st.error(f"⚠️ {detail['error']}")
            if detail.get("unknown"):
                st.caption(f"Unrecognized: {', '.join(detail['unknown'])}")
        elif detail.get("insufficient_information"):
            st.warning(f"ℹ️ {detail['message']}")
        st.stop()

    if resp.status_code != 200:
        st.error(f"⚠️ Backend error: {resp.text}")
        st.stop()

    result = resp.json()

    if result.get("red_flag"):
        st.markdown(f"""
        <div class="disclaimer">🚨 <strong>URGENT:</strong> {result['message']}<br>
        Triggered by: {", ".join(result['triggered_by'])}</div>
        """, unsafe_allow_html=True)
        st.stop()

    top = result["top_prediction"]
    top3 = result["top3"]

    if result["low_confidence"]:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Low confidence prediction.</strong> 
            The symptoms provided are ambiguous. Please consult a doctor for proper diagnosis.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🎯 Prediction Results")
    st.caption(result.get("confidence_note", ""))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🦠 Predicted Disease", top["disease"])
    with col2:
        st.metric("📊 Confidence", f"{top['confidence_pct']}%")
    with col3:
        st.metric("👨‍⚕️ Recommended Specialist", top["doctor"])

    st.caption(f"Margin over 2nd choice: {result['margin']*100:.1f} pts · Uncertainty: {result['uncertainty']*100:.1f}%")

    st.markdown("**Confidence Level**")
    st.progress(min(max(top["confidence"], 0.0), 1.0))

    st.markdown("---")

    st.markdown("### 📋 Differential Diagnosis (Top 3)")

    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (pred, col) in enumerate(zip(top3, cols)):
        with col:
            with st.container():
                st.markdown(f"**{medals[i]} #{i+1}**")
                st.markdown(f"**{pred['disease']}**")
                st.caption(f"Confidence: {pred['confidence_pct']}%")
                st.progress(min(max(pred["confidence"], 0.0), 1.0))
                st.caption(f"👨‍⚕️ {pred['doctor']}")

    st.markdown("---")

    st.markdown("### 🔍 AI Explainability (SHAP)")
    st.caption("SHAP impact scores — not probabilities. Positive pushes toward this disease.")

    with st.spinner("Computing SHAP values..."):
        explain_resp = api_post("/explain", {
            "symptoms": result["matched_symptoms"],
            "predicted_disease": top["disease"],
        })
        factors = explain_resp.json().get("top_factors", []) if explain_resp.status_code == 200 else []

    if factors:
        for factor in factors[:6]:
            is_positive = factor["impact_value"] > 0
            color = "#1a73e8" if is_positive else "#ea4335"
            bar_width = min(abs(factor["impact_value"]) * 300, 100)
            present_badge = "✅ Present" if factor["present"] else "⬜ Absent"
            
            col_name, col_bar, col_val, col_present = st.columns([2, 3, 1, 1])
            col_name.markdown(f"**{factor['symptom']}**")
            col_bar.markdown(
                f'<div style="background:#f0f0f0;border-radius:4px;height:20px;margin-top:3px;">'
                f'<div style="background:{color};width:{bar_width}%;height:100%;border-radius:4px;"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            col_val.markdown(f"**{factor['impact_label']}** ({factor['strength']})")
            col_present.markdown(present_badge)

    st.markdown("---")

    st.markdown("### 🗣️ Plain-Language Summary")
    try:
        with st.spinner("Generating summary..."):
            llm_resp = api_post("/explain-plain-language", {"symptoms": selected_symptoms})
        if llm_resp.status_code == 200:
            llm_data = llm_resp.json()
            st.info(llm_data["explanation"])
            if llm_data["source"] != "llm":
                st.caption("Generated from a template (no LLM API key configured).")
        else:
            st.warning("Plain-language summary is unavailable for this prediction.")
    except requests.RequestException:
        st.warning("Plain-language summary is unavailable because the backend could not be reached.")

    st.markdown("---")

    # ── PDF Report ─────────────────────────────────────────────────────────────
    st.markdown("### 📄 Download Report")

    with st.spinner("Generating PDF..."):
        report_resp = api_post("/report", {
            "symptoms": selected_symptoms,
            "patient_name": patient_name or "Patient",
        })
        if report_resp.status_code == 200:
            st.download_button(
                label="📥 Download PDF Report",
                data=report_resp.content,
                file_name="medpredict_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.error("PDF generation failed.")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ DISCLAIMER: This AI prediction is for educational purposes only. '
        'Always consult a qualified healthcare professional for medical advice and diagnosis.</div>',
        unsafe_allow_html=True
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.85rem;'>"
    "MedPredict AI v2 • Built with XGBoost, SHAP, FastAPI & Streamlit • "
    "For educational purposes only"
    "</div>",
    unsafe_allow_html=True
)
