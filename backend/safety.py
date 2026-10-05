"""
MedPredict AI - Red-Flag Safety Layer
Deterministic rule check, run BEFORE the ML model. Never a prediction —
this is a safety gate, so it stays simple, explainable, and rule-based.
"""

# Trigger urgent-care messaging on their own
SINGLE_RED_FLAGS = {
    "difficulty_breathing": "Severe difficulty breathing",
    "blood_in_cough": "Coughing up blood",
    "loss_of_consciousness": "Loss of consciousness",
}

# Trigger only when BOTH symptoms in a pair are present
COMBINATION_RED_FLAGS = [
    ({"chest_pain", "shortness_of_breath"}, "Chest pain with shortness of breath"),
    ({"severe_headache", "vomiting"}, "Severe headache with vomiting"),
    ({"fever", "chest_pain", "shortness_of_breath"}, "High fever with breathing difficulty"),
]

URGENT_MESSAGE = (
    "These symptoms may indicate a medical emergency. Please seek immediate "
    "medical attention or contact emergency services — do not rely on this "
    "tool's prediction."
)


def check_red_flags(raw_user_symptoms: list[str]) -> dict | None:
    """Runs on normalized raw input, independent of the model's known vocabulary."""
    normalized = {s.lower().strip().replace(" ", "_") for s in raw_user_symptoms}

    triggered = []
    for symptom, label in SINGLE_RED_FLAGS.items():
        if symptom in normalized:
            triggered.append(label)

    for combo, label in COMBINATION_RED_FLAGS:
        if combo.issubset(normalized):
            triggered.append(label)

    if not triggered:
        return None

    return {
        "red_flag": True,
        "triggered_by": triggered,
        "message": URGENT_MESSAGE,
    }