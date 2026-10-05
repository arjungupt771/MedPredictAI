"""
MedPredict AI - SHAP Explainability (v2.1)
Explains WHY a prediction was made. SHAP values are reported as raw impact
scores + direction, NOT as percentages — a SHAP value is not a probability.
"""
import shap
import numpy as np
from predictor import symptoms_to_vector, get_all_symptoms, _model, _le

_symptoms = get_all_symptoms()
_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(_model)
    return _explainer


def _impact_strength(abs_value: float) -> str:
    if abs_value >= 0.15:
        return "Strong"
    if abs_value >= 0.07:
        return "Moderate"
    return "Weak"


def get_explanation(recognized_symptoms: list[str], predicted_disease: str) -> dict:
    vector = symptoms_to_vector(recognized_symptoms)
    explainer = _get_explainer()
    shap_values = explainer.shap_values(np.array([vector]))

    classes = list(_le.classes_)
    class_idx = classes.index(predicted_disease) if predicted_disease in classes else 0

    if isinstance(shap_values, list):
        sv = np.array(shap_values[class_idx][0], dtype=float)
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv = np.array(shap_values[0, :, class_idx], dtype=float)
    else:
        sv = np.array(shap_values[0], dtype=float)

    symptom_shap = [
        {"symptom": sym, "shap_value": float(sv[i]), "present": vector[i] == 1}
        for i, sym in enumerate(_symptoms)
    ]
    symptom_shap.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_factors = [s for s in symptom_shap if abs(s["shap_value"]) > 1e-4][:10]

    formatted = []
    for item in top_factors:
        direction = "Supports prediction" if item["shap_value"] > 0 else "Against prediction"
        formatted.append({
            "symptom": item["symptom"].replace("_", " ").title(),
            "impact_value": round(item["shap_value"], 4),
            "impact_label": f"{'+' if item['shap_value'] > 0 else '-'}{abs(round(item['shap_value'], 3))}",
            "direction": direction,
            "strength": _impact_strength(abs(item["shap_value"])),
            "present": item["present"],
        })

    return {
        "disease": predicted_disease,
        "top_factors": formatted[:8],
        "note": (
            "Values are SHAP impact scores, not probabilities. Positive values "
            "push the prediction toward this disease; negative values push away from it."
        ),
    }