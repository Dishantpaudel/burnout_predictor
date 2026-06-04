"""
FastAPI serving endpoint for the Stress Level Predictor.

Run with:
    uvicorn serving.main:app --reload

Then open:
    http://localhost:8000/docs   <- Swagger UI (live demo)
    http://localhost:8000/health <- health check
"""

import os
import sys
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from serving.schemas import LifestyleInput, PredictionOutput

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stress Level Predictor",
    description="Predicts stress level (Low / Moderate / High) from daily lifestyle habits.",
    version="1.0.0",
)

# ── Load model artifacts once at startup ─────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'model')

model        = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
features     = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
target_names = joblib.load(os.path.join(MODEL_DIR, 'target_names.pkl'))
encoders     = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))


# ── Helper ────────────────────────────────────────────────────────────────────
def build_feature_row(data: LifestyleInput) -> list:
    exercise_map = {"Low": 0, "Moderate": 1, "High": 2}

    try:
        gender_enc  = int(encoders["le_gender"].transform([data.gender])[0])
        diet_enc    = int(encoders["le_diet"].transform([data.diet_type])[0])
        country_enc = int(encoders["le_country"].transform([data.country])[0])
        cond_enc    = int(encoders["le_cond"].transform([data.mental_health_condition])[0])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Unknown category value: {e}")

    if data.exercise_level not in exercise_map:
        raise HTTPException(status_code=422, detail=f"exercise_level must be Low, Moderate, or High")

    sleep_debt      = max(0, 8 - data.sleep_hours)
    screen_overload = int(data.screen_time_hours > 6)
    work_intensity  = data.work_hours_per_week / 40

    return [
        data.age,
        data.sleep_hours,
        data.work_hours_per_week,
        data.screen_time_hours,
        data.social_interaction_score,
        data.happiness_score,
        sleep_debt,
        screen_overload,
        work_intensity,
        exercise_map[data.exercise_level],
        gender_enc,
        diet_enc,
        country_enc,
        cond_enc,
    ]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "target": "Stress Level"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: LifestyleInput):
    """
    Predict stress level from lifestyle habits.

    - **Low** — lifestyle patterns associated with low stress
    - **Moderate** — lifestyle patterns associated with moderate stress
    - **High** — lifestyle patterns associated with high stress
    """
    row    = build_feature_row(data)
    pred   = int(model.predict([row])[0])
    probas = model.predict_proba([row])[0]

    return PredictionOutput(
        predicted_stress_level=target_names[pred],
        probabilities={name: round(float(p), 4) for name, p in zip(target_names, probas)}
    )
