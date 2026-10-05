"""
MedPredict AI - FastAPI Backend (v2.1)
"""
import json
import platform
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from predictor import predict_disease, get_all_symptoms
from explain import get_explanation
from report import generate_report
from experiment_logger import list_experiments, best_experiment
from llm_explainer import generate_plain_language_explanation

MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="MedPredict AI",
    description="AI-powered disease prediction with explainability and PDF reports",
    version="2.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    symptoms: list[str] = Field(..., example=["fever", "cough", "headache"])
    patient_name: Optional[str] = Field(default="Patient")

class ExplainRequest(BaseModel):
    symptoms: list[str]
    predicted_disease: str

class ReportRequest(BaseModel):
    symptoms: list[str]
    patient_name: Optional[str] = "Patient"

class LLMExplainRequest(BaseModel):
    symptoms: list[str]


@app.get("/")
def root():
    return {
        "app": "MedPredict AI",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/explain", "/explain-plain-language", "/experiments", "/symptoms", "/model-stats", "/report", "/health", "/version", "/model-info"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"app_version": "2.1.0", "python_version": sys.version.split()[0], "platform": platform.platform()}


@app.get("/model-info")
def model_info():
    meta_path = MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"error": "Model metadata not available. Re-run training to generate it."}


@app.get("/symptoms")
def list_symptoms():
    symptoms = get_all_symptoms()
    return {
        "count": len(symptoms),
        "symptoms": symptoms,
        "display": [s.replace("_", " ").title() for s in symptoms],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom required")

    result = predict_disease(req.symptoms)

    # Validation failures → 422 so clients can distinguish "bad input" from
    # a normal (if uncertain) prediction result.
    if "error" in result:
        raise HTTPException(status_code=422, detail=result)

    return result


@app.post("/explain")
def explain(req: ExplainRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms required for explanation")
    try:
        return get_explanation(req.symptoms, req.predicted_disease)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/explain-plain-language")
def explain_plain_language(req: LLMExplainRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms required")

    pred_result = predict_disease(req.symptoms)
    if "error" in pred_result or "red_flag" in pred_result or "insufficient_information" in pred_result:
        raise HTTPException(status_code=422, detail=pred_result)

    top = pred_result["top_prediction"]
    shap_result = get_explanation(pred_result["matched_symptoms"], top["disease"])
    return generate_plain_language_explanation(top, shap_result["top_factors"])


@app.post("/report")
def generate(req: ReportRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms required")

    pred_result = predict_disease(req.symptoms)
    if "error" in pred_result or "red_flag" in pred_result or "insufficient_information" in pred_result:
        raise HTTPException(status_code=422, detail=pred_result)

    top_pred = pred_result["top_prediction"]
    top3 = pred_result["top3"]

    try:
        shap_result = get_explanation(pred_result["matched_symptoms"], top_pred["disease"])
        factors = shap_result.get("top_factors", [])
    except Exception:
        factors = []

    pdf_bytes = generate_report(
        symptoms=req.symptoms,
        top_prediction=top_pred,
        top3=top3,
        shap_factors=factors,
        patient_name=req.patient_name or "Patient",
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=medpredict_report.pdf"},
    )


@app.get("/model-stats")
def model_stats():
    stats_path = MODEL_DIR / "model_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {"error": "Stats not available"}


@app.get("/experiments")
def experiments():
    return {"runs": list_experiments(), "best": best_experiment()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)