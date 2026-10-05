import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))
from validation import validate_symptoms

def test_empty_symptoms_rejected():
    result = validate_symptoms([])
    assert result["valid"] is False

def test_all_unknown_rejected():
    result = validate_symptoms(["xyz_nonexistent"])
    assert result["valid"] is False
    assert "xyz_nonexistent" in result["unknown"]

def test_duplicates_deduped():
    result = validate_symptoms(["fever", "Fever", "fever "])
    assert result["recognized"].count("fever") == 1

def test_single_symptom_is_insufficient():
    result = validate_symptoms(["fever"])
    assert result["valid"] is True
    assert result["sufficient"] is False

def test_multiple_recognized_is_sufficient():
    result = validate_symptoms(["fever", "cough", "headache"])
    assert result["sufficient"] is True