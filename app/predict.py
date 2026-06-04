import numpy as np
import pandas as pd
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

_model = None
_le_target = None
_encoders = None
_features = None


def _load():
    global _model, _le_target, _encoders, _features
    if _model is None:
        _model = joblib.load(f'{MODEL_DIR}/model.pkl')
        _le_target = joblib.load(f'{MODEL_DIR}/label_encoder.pkl')
        _encoders = joblib.load(f'{MODEL_DIR}/encoders.pkl')
        _features = joblib.load(f'{MODEL_DIR}/feature_names.pkl')


def predict(
    age, sleep_hours, work_hours, screen_time,
    social_score, happiness_score,
    exercise_level, stress_level,
    gender, diet, country
):
    _load()

    exercise_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}

    gender_enc = _encoders['le_gender'].transform([gender])[0]
    diet_enc = _encoders['le_diet'].transform([diet])[0]
    country_enc = _encoders['le_country'].transform([country])[0]

    sleep_debt = max(0, 8 - sleep_hours)
    screen_overload = int(screen_time > 6)
    work_intensity = work_hours / 40

    row = pd.DataFrame([[
        age, sleep_hours, work_hours, screen_time,
        social_score, happiness_score,
        sleep_debt, screen_overload, work_intensity,
        exercise_map[exercise_level], stress_map[stress_level],
        gender_enc, diet_enc, country_enc
    ]], columns=_features)

    pred_idx = _model.predict(row)[0]
    probas = _model.predict_proba(row)[0]
    label = _le_target.classes_[pred_idx]
    proba_dict = dict(zip(_le_target.classes_, probas.round(4)))

    return label, proba_dict
