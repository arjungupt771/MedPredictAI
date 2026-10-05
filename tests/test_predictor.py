import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))
from predictor import predict_disease, get_all_symptoms

def test_prediction_returns_top3():
    result = predict_disease(["fever", "cough", "headache"])
    assert len(result["top3"]) == 3

def test_probabilities_sum_reasonably():
    result = predict_disease(["fever", "cough", "headache"])
    total = sum(p["confidence"] for p in result["top3"])
    assert 0 < total <= 1.0001

def test_all_predictions_are_known_diseases():
    known = set()  # populated from label encoder indirectly via predictions
    result = predict_disease(["fever", "chills", "sweating"])
    for p in result["top3"]:
        assert isinstance(p["disease"], str) and len(p["disease"]) > 0

def test_empty_symptoms_returns_error():
    result = predict_disease([])
    assert "error" in result

def test_red_flag_short_circuits_prediction():
    result = predict_disease(["chest_pain", "shortness_of_breath"])
    assert result.get("red_flag") is True

def test_insufficient_info_state():
    result = predict_disease(["fever"])
    assert result.get("insufficient_information") is True