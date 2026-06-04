# Project Roadmap — End-to-End ML Pipeline

**Topic:** Predicting Mental Health Conditions from Daily Lifestyle Habits
**Task:** Multi-class Classification (5 classes)
**Grade weight:** 40% of final grade (10 points)
**Deadline:** Submit repo 2 hours before Friday presentation

---

## The 3 Deliverables (what the teacher grades)

```
01 NOTEBOOK        02 MODELING            03 SERVING
(4 points)         (3 points)             (3 points)
─────────────      ─────────────────      ──────────────────
EDA                Modular code           FastAPI app
Baseline           Config file            Pydantic schemas
+ 1 improvement    MLflow tracking        Swagger UI demo
Feature importance Optuna tuning          README
Interpretation     DVC + GDrive
Teacher discussion README
```

**Key rule:** Teacher must be able to clone the repo and reproduce your best run with zero questions.

---

## Decisions Already Locked In

| Decision | Choice | Why |
|---|---|---|
| Baseline model | Decision Tree | Simple single tree — no linear regression (teacher said skip) |
| Improvement model | XGBoost + Optuna | Boosting ensemble, tuned — clear "better" story |
| Metric | F1 Macro (+ accuracy) | 5 classes, fairness across rare conditions |
| Feature importance | SHAP | Explains *why*, not just *what* |
| Serving | FastAPI (not Streamlit) | Assignment requires it |

---

## THE STEPS (where we are + what's next)

### PHASE 0 — Setup & Git
- [ ] **0.1** Initialize git repo + `.gitignore` (ignore `mlruns/`, data, models)
- [ ] **0.2** Final folder structure
- [ ] **0.3** `requirements.txt` updated (add mlflow, optuna, fastapi, dvc, pyyaml)
- [ ] **0.4** `config.yaml` skeleton

### PHASE 1 — Notebook (Part 01)  ← teacher reviews this FIRST
- [x] 1.1 Load data + understand columns
- [x] 1.2 EDA plots
- [ ] **1.3** EDA that drives a REAL decision (must influence modelling) ← needs strengthening
- [x] 1.4 Feature engineering
- [ ] **1.5** Baseline = Decision Tree (remove Logistic Regression)
- [ ] **1.6** Improvement = XGBoost, explain WHY it's better
- [x] 1.7 SHAP feature importance
- [ ] **1.8** Plain-language interpretation commentary
- [ ] **1.9** Share notebook with teacher for 10-min discussion

### PHASE 2 — MLOps Modular Pipeline (Part 02)
- [ ] **2.1** `config.yaml` — all params, paths, hyperparams
- [ ] **2.2** Modularise: `data.py`, `features.py`, `train.py`, `evaluate.py`
- [ ] **2.3** Proper train/val/test split (no leakage)
- [ ] **2.4** Optuna hyperparameter tuning loop
- [ ] **2.5** MLflow experiment tracking (local UI)
- [ ] **2.6** DVC init + Google Drive remote
- [ ] **2.7** Share GDrive folder with aaaksenova2@gmail.com (Viewer)
- [ ] **2.8** README for reproducibility

### PHASE 3 — FastAPI Serving (Part 03)
- [ ] **3.1** Pydantic input schema (lifestyle features)
- [ ] **3.2** Pydantic output schema (prediction + probabilities)
- [ ] **3.3** `/predict` endpoint that loads model
- [ ] **3.4** Test in Swagger UI (`/docs`)
- [ ] **3.5** Serving README

### PHASE 4 — Presentation (Friday)
- [ ] **4.1** Slides covering all 8 required questions
- [ ] **4.2** Production reasoning (batch/streaming, drift, online metrics)
- [ ] **4.3** Live demo: one query through FastAPI

---

## The 8 Presentation Questions (must answer all)

1. Why is this problem worth solving?
2. The data: size, features, problems, processing decisions?
3. Which parameters / hyperparameters matter and why?
4. What metric did you choose and why?
5. Model choice: tradeoffs? Baseline vs improvement? Final metrics?
6. How would this work in production? (batch / streaming)
7. What online metrics and data drift would you monitor?
8. Live demo through FastAPI

---

## Current Status

> **WE ARE HERE:** About to start **Phase 0 (Setup)**, then redo **Phase 1** properly.
>
> What exists from before: basic notebook, basic src files, Streamlit predict.py.
> What changes: remove Logistic Regression, switch Streamlit → FastAPI, add MLflow + Optuna + DVC + config.

---

## How We Work (your rule)

- **Opus** = planning & understanding (now)
- **Sonnet** = implementing one step at a time
- After each step: I tell you ✅ what we did, ➡️ what's next
- You learn each piece before moving on
