"""
MedPredict AI - Input Validation
Normalizes and validates raw symptom input before it reaches the model.
"""
from predictor import get_all_symptoms

MIN_RECOGNIZED_SYMPTOMS = 2  # below this → "insufficient information", not a prediction


def normalize(symptom: str) -> str:
    return symptom.lower().strip().replace(" ", "_")


def validate_symptoms(user_symptoms: list[str]) -> dict:
    """
    Returns:
        {
          "valid": bool,
          "error": str | None,
          "recognized": [...],   # normalized, deduped, known symptoms
          "unknown": [...],      # normalized symptoms not in the vocabulary
          "sufficient": bool,    # enough signal for a prediction
        }
    """
    known = set(get_all_symptoms())

    if not user_symptoms:
        return {
            "valid": False,
            "error": "No symptoms were provided.",
            "recognized": [], "unknown": [], "sufficient": False,
        }

    normalized = [normalize(s) for s in user_symptoms if normalize(s)]
    deduped = list(dict.fromkeys(normalized))  # preserve order, drop dupes

    recognized = [s for s in deduped if s in known]
    unknown = [s for s in deduped if s not in known]

    if not recognized:
        return {
            "valid": False,
            "error": "No recognized symptoms were provided.",
            "recognized": [], "unknown": unknown, "sufficient": False,
        }

    return {
        "valid": True,
        "error": None,
        "recognized": recognized,
        "unknown": unknown,
        "sufficient": len(recognized) >= MIN_RECOGNIZED_SYMPTOMS,
    }