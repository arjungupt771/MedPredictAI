"""
MedPredict AI - Prediction Engine (v2.1)
Calibrated confidence + uncertainty, input validation, red-flag safety gate,
and an explicit "insufficient information" state instead of always predicting.
"""
import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

# _model: raw tree ensemble, used ONLY for SHAP (explain.py)
# _calibrated_model: sigmoid-calibrated wrapper, used for actual confidence scores
_model = joblib.load(MODEL_DIR / "model.pkl")
_calibrated_model = joblib.load(MODEL_DIR / "calibrated_model.pkl")
_le = joblib.load(MODEL_DIR / "label_encoder.pkl")
_symptoms: list[str] = joblib.load(MODEL_DIR / "symptoms.pkl")

DOCTOR_MAP = {
    "Influenza": "General Physician", "Common Cold": "General Physician",
    "COVID-19": "Infectious Disease Specialist", "Pneumonia": "Pulmonologist",
    "Bronchitis": "Pulmonologist", "Malaria": "Infectious Disease Specialist",
    "Dengue": "Infectious Disease Specialist", "Typhoid": "Infectious Disease Specialist",
    "Jaundice": "Gastroenterologist / Hepatologist", "Diabetes": "Endocrinologist",
    "Hypertension": "Cardiologist", "Gastroenteritis": "Gastroenterologist",
    "Migraine": "Neurologist", "Asthma": "Pulmonologist / Allergist",
    "Appendicitis": "General Surgeon (Emergency)", "UTI": "Urologist / General Physician",
    "Chickenpox": "Dermatologist / General Physician", "Measles": "General Physician / Pediatrician",
    "Tuberculosis": "Pulmonologist", "Anemia": "Hematologist",
}

LOW_CONFIDENCE_THRESHOLD = 0.60


def get_all_symptoms() -> list[str]:
    return _symptoms


def symptoms_to_vector(recognized_symptoms: list[str]) -> list[int]:
    """Assumes input is already normalized + validated (see validation.py)."""
    vector = [0] * len(_symptoms)
    for s in recognized_symptoms:
        if s in _symptoms:
            vector[_symptoms.index(s)] = 1
    return vector


def predict_disease(user_symptoms: list[str]) -> dict:
    """
    Full pipeline: validate -> red-flag check -> insufficient-info check -> predict.
    Returns exactly one of: {"error": ...}, {"red_flag": ...}, {"insufficient_information": ...},
    or a normal prediction payload.
    """
    from validation import validate_symptoms
    from safety import check_red_flags

    validation = validate_symptoms(user_symptoms)
    if not validation["valid"]:
        return {
            "error": validation["error"],
            "recognized": validation["recognized"],
            "unknown": validation["unknown"],
        }

    # Safety gate runs on raw input, before any ML — deterministic, not a prediction.
    red_flag = check_red_flags(user_symptoms)
    if red_flag:
        return red_flag

    if not validation["sufficient"]:
        return {
            "insufficient_information": True,
            "message": (
                "Only limited symptom information was provided. Please add more "
                "symptoms before generating a prediction."
            ),
            "recognized": validation["recognized"],
            "unknown": validation["unknown"],
        }

    recognized = validation["recognized"]
    vector = symptoms_to_vector(recognized)
    proba = _calibrated_model.predict_proba([vector])[0]

    order = np.argsort(proba)[::-1]
    top3_idx = order[:3]
    top3 = []
    for idx in top3_idx:
        disease = _le.inverse_transform([idx])[0]
        confidence = round(float(proba[idx]), 4)
        top3.append({
            "disease": disease,
            "confidence": confidence,
            "confidence_pct": round(confidence * 100, 1),
            "doctor": DOCTOR_MAP.get(disease, "General Physician"),
        })

    best = top3[0]
    second = top3[1]["confidence"] if len(top3) > 1 else 0.0
    margin = round(best["confidence"] - second, 4)
    uncertainty = round(1.0 - best["confidence"], 4)
    low_confidence = best["confidence"] < LOW_CONFIDENCE_THRESHOLD

    return {
        "top_prediction": best,
        "top3": top3,
        "margin": margin,
        "uncertainty": uncertainty,
        "low_confidence": low_confidence,
        "message": "Consult a doctor regardless for professional diagnosis." if low_confidence else None,
        "matched_symptoms": recognized,
        "unrecognized_symptoms": validation["unknown"],
        "confidence_note": "Confidence is a calibrated model probability, not diagnostic certainty.",
    }


if __name__ == "__main__":
    result = predict_disease(["fever", "cough", "headache"])
    print("Test prediction:", result.get("top_prediction"))