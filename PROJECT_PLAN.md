# Mental Health Condition Predictor — Project Plan

**Problem:** Predict which mental health condition a person's lifestyle is linked to  
**Dataset:** [Mental Health and Lifestyle Habits (2019–2024) — Kaggle](https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024)  
**Task:** Multi-class Classification — 5 classes: `None`, `Anxiety`, `Depression`, `PTSD`, `Bipolar`  
**Data:** 3,001 rows × 12 columns  
**Due:** Friday presentation

---

## Project Structure

```
burnout_predictor/
│
├── PROJECT_PLAN.md                     ← this file
├── requirements.txt                    ← all Python packages
│
├── data/
│   └── raw/
│       └── Mental_Health_Lifestyle_Dataset.csv   ← downloaded dataset
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb       ← Phase 1: full EDA + model training
│
├── src/
│   ├── preprocess.py                   ← data loading + feature engineering
│   └── train.py                        ← training script (MLOps style)
│
└── app/
    ├── predict.py                      ← prediction logic
    ├── main.py                         ← Streamlit demo app  ← NOT YET BUILT
    └── model/                          ← saved model files (created after training)
        ├── model.pkl
        ├── label_encoder.pkl
        ├── feature_names.pkl
        └── encoders.pkl
```

---

## Dataset Columns

| Column | Type | Description |
|---|---|---|
| Country | Categorical | 8 countries (Brazil, USA, India…) |
| Age | Numeric | 18–65 |
| Gender | Categorical | Male, Female, Other |
| Exercise Level | Categorical | Low, Moderate, High |
| Diet Type | Categorical | Balanced, Vegan, Vegetarian, Keto, Junk Food |
| Sleep Hours | Numeric | Hours of sleep per day |
| Stress Level | Categorical | Low, Moderate, High |
| Work Hours per Week | Numeric | Hours worked per week |
| Screen Time per Day (Hours) | Numeric | Daily screen time |
| Social Interaction Score | Numeric | 1–10 scale |
| Happiness Score | Numeric | 1–10 scale |
| **Mental Health Condition** | **Target** | **None, Anxiety, Depression, PTSD, Bipolar** |

---

## Build Steps (do these in order)

### Step 1 — Install dependencies ✅ (do this first)
```bash
pip install -r requirements.txt
```

### Step 2 — Run the EDA Notebook (Phase 1)
Open `notebooks/01_eda_and_modeling.ipynb` and run cell by cell.

What you will do inside the notebook:
1. Load the CSV and understand the data
2. EDA — plots, distributions, correlations
3. Feature Engineering — create `sleep_debt`, `screen_overload`, `work_intensity`
4. Baseline model — Logistic Regression
5. Ensemble model 1 — Random Forest (bagging)
6. Ensemble model 2 — XGBoost (boosting)
7. Model comparison table
8. SHAP explainability — why does the model predict what it predicts?
9. Save the best model to `app/model/`

### Step 3 — Build Streamlit App (Phase 2)
File: `app/main.py` — **not built yet, we do this after Step 2**

The app lets someone fill in their lifestyle habits and get a prediction:
- Sliders for sleep hours, screen time, stress level, etc.
- Output: predicted condition + probability bar chart + top 3 SHAP factors

### Step 4 — Write README
A short description of the project for your GitHub / CV.

### Step 5 — Presentation (Friday)
5 slides — see outline below.

---

## Models We Train (and Why)

| Model | Type | Why We Use It |
|---|---|---|
| Logistic Regression | Baseline (linear) | Simple starting point to beat |
| Random Forest | Bagging ensemble | Many trees vote together — robust |
| XGBoost | Boosting ensemble | Trees learn from each other's mistakes — usually best |

---

## Engineered Features (created in notebook)

| Feature | Formula | What It Captures |
|---|---|---|
| `sleep_debt` | `max(0, 8 - sleep_hours)` | How far below recommended 8h |
| `screen_overload` | `screen_time > 6` → 0 or 1 | Binary flag for high screen time |
| `work_intensity` | `work_hours / 40` | Work load relative to standard week |

---

## Current Status

| Task | Status |
|---|---|
| Dataset downloaded | ✅ Done |
| requirements.txt | ✅ Done |
| notebooks/01_eda_and_modeling.ipynb | ✅ Done (ready to run) |
| src/preprocess.py | ✅ Done |
| src/train.py | ✅ Done |
| app/predict.py | ✅ Done |
| app/main.py (Streamlit app) | ⏳ After notebook is run |
| Model trained + saved | ⏳ After notebook is run |
| README.md | ⏳ Last step |
| Friday presentation | ⏳ Last step |

---

## Presentation Outline (5 slides)

| Slide | Content |
|---|---|
| 1 | **The Problem** — burnout/mental health stats, why it matters |
| 2 | **The Data** — what features we have, 2–3 EDA charts |
| 3 | **The Models** — baseline vs ensemble, comparison table |
| 4 | **Live Demo** — Streamlit app + SHAP explanation chart |
| 5 | **Takeaways** — what worked, what you'd improve |

---

## What Makes This CV-Worthy

- **Real problem** — mental health is a global priority, not a toy dataset
- **Full pipeline** — raw CSV → trained model → live interactive demo
- **Explainable AI (SHAP)** — shows you understand *why* the model predicts, not just *what*
- **MLOps structure** — modular code (`src/`, `app/`) like a real team uses
- **Deployable** — Streamlit app can be shared as a free link on Streamlit Cloud
