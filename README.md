# Stress Level Predictor

Predicts a person's stress level (Low / Moderate / High) from daily lifestyle habits using XGBoost with Optuna hyperparameter tuning, MLflow experiment tracking, and DVC artifact versioning.

---

## Problem

Can we predict whether someone is at Low, Moderate, or High stress risk based on their sleep patterns, work hours, screen time, exercise habits, and other lifestyle factors?

**Dataset:** Mental Health and Lifestyle Habits (2019-2024) — 3,000 records, 12 features  
**Task:** Multi-class classification (3 classes)  
**Metric:** F1 Macro (treats all 3 classes equally)

---

## Results

| Model | F1 Macro | Accuracy |
|---|---|---|
| Decision Tree (baseline) | 0.2819 | 0.3200 |
| XGBoost default | 0.3411 | 0.3417 |
| **XGBoost + Optuna tuned** | **0.3669** | **0.3683** |

Best hyperparameters found by Optuna:
- `n_estimators`: 235
- `learning_rate`: 0.016
- `max_depth`: 5
- `subsample`: 0.96
- `colsample_bytree`: 0.98

---

## Project Structure

```
burnout_predictor/
├── config.yaml               # all paths, features, model params
├── requirements.txt          # dependencies
├── README.md
│
├── data/raw/                 # tracked by DVC (not in git)
│   └── Mental_Health_Lifestyle_Dataset.csv
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # EDA + baseline + XGBoost + SHAP
│
├── src/
│   ├── data.py               # load and clean data
│   ├── features.py           # feature engineering + encoding
│   ├── train.py              # MLflow + Optuna training pipeline
│   └── evaluate.py           # metrics and SHAP plots
│
├── serving/
│   ├── main.py               # FastAPI /predict endpoint
│   ├── schemas.py            # Pydantic input/output schemas
│   └── README.md             # how to run the API
│
└── app/model/                # tracked by DVC (not in git)
    ├── model.pkl
    ├── label_encoder.pkl
    ├── feature_names.pkl
    ├── target_names.pkl
    └── encoders.pkl
```

---

## Reproduce from Scratch

### 1. Clone and install

```bash
git clone <your-repo-url>
cd burnout_predictor
pip install -r requirements.txt
```

### 2. Pull data and model from DVC

```bash
dvc pull
```

> Requires Google Drive access. DVC remote is configured in `.dvc/config`.

### 3. Run the full training pipeline

```bash
python src/train.py
```

This will:
- Load and clean data
- Engineer features
- Run 50 Optuna trials to find best hyperparameters
- Log all experiments to MLflow
- Train final model and save to `app/model/`

### 4. View experiment results

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser.

### 5. Start the prediction API

```bash
uvicorn serving.main:app --reload
```

Open `http://localhost:8000/docs` to see the Swagger UI and test the endpoint.

---

## API Usage

Send a POST request to `/predict`:

```json
{
  "age": 30,
  "sleep_hours": 5.5,
  "work_hours_per_week": 55,
  "screen_time_hours": 7.0,
  "social_interaction_score": 4.0,
  "happiness_score": 3.5,
  "exercise_level": "Low",
  "gender": "Male",
  "diet_type": "Junk Food",
  "country": "USA",
  "mental_health_condition": "None"
}
```

Response:

```json
{
  "predicted_stress_level": "High",
  "probabilities": {
    "Low": 0.12,
    "Moderate": 0.23,
    "High": 0.65
  }
}
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Main prediction model |
| Optuna | Hyperparameter tuning (50 trials) |
| MLflow | Experiment tracking |
| DVC | Data and model versioning |
| SHAP | Model explainability |
| FastAPI | Model serving endpoint |
| scikit-learn | Preprocessing and metrics |
