import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_symptoms_endpoint():
    r = client.get("/symptoms")
    assert r.status_code == 200
    assert r.json()["count"] > 0

def test_predict_valid():
    r = client.post("/predict", json={"symptoms": ["fever", "cough", "headache"]})
    assert r.status_code == 200
    assert "top_prediction" in r.json()

def test_predict_unknown_symptoms_rejected():
    r = client.post("/predict", json={"symptoms": ["not_a_real_symptom"]})
    assert r.status_code == 422

def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200