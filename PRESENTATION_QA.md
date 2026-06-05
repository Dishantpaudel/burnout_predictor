# Presentation Q&A — Stress Level Predictor

---

## Q1. Why is this problem worth solving?

Stress is one of the leading causes of burnout, mental health problems, and reduced productivity worldwide. The dangerous thing about stress is that most people don't realise they are heading toward a crisis until it's already serious.

The idea behind this project is simple: if someone shares their daily habits — how much they sleep, how many hours they work, how much screen time they have, whether they exercise — can we predict their stress level before it becomes a problem?

This has real applications. HR departments can use it to flag employees at risk. Health apps can use it to give personalised lifestyle recommendations. It is a preventative tool, not a reactive one.

---

## Q2. What is the data? Size, features, problems, processing decisions?

The dataset has 3,000 records and 12 columns, collected from people across 7 countries.

**Features include:**
- Numeric: Age, Sleep Hours, Work Hours per Week, Screen Time per Day, Social Interaction Score, Happiness Score
- Ordinal: Exercise Level (Low / Moderate / High), Stress Level
- Categorical: Gender, Diet Type, Country, Mental Health Condition

**Problem found in EDA:**
595 rows had missing values in Mental Health Condition. These were not errors — they represented healthy people with no diagnosis. Decision: fill with "None" so we don't lose 20% of the data.

**Why we changed the target variable:**
We originally planned to predict Mental Health Condition (5 classes: Anxiety, Depression, Bipolar, PTSD, None). During EDA, we found that all 5 classes had nearly identical feature distributions — every model scored around 20% F1, which is the same as random guessing for 5 balanced classes. This told us the lifestyle features in this dataset do not separate clinical diagnoses.

Stress Level, on the other hand, has a direct and well-established causal link to daily habits — less sleep means more stress, more work means more stress. So we switched the target to Stress Level (Low / Moderate / High), which gave meaningful, learnable signal.

---

## Q3. Which hyperparameters matter and why?

For XGBoost there are three key hyperparameters:

**learning_rate** — controls how much each new tree corrects the previous tree's mistakes. A lower value means the model is more careful and conservative per step. Default is 0.1. Optuna found 0.016 was better — much more careful corrections, less overfitting.

**n_estimators** — how many trees to build. More trees generally means better performance but slower training. Optuna found 235 was the sweet spot.

**max_depth** — how deep each individual tree grows. Deeper trees learn more complex patterns but can overfit. We used max_depth=5 throughout.

Optuna tested 50 different combinations of these parameters automatically, optimising for F1 Macro on the validation set. The best combination improved F1 from 0.34 to 0.37.

---

## Q4. What metric did you choose and why?

We use **F1 Macro**.

Here is why we did not just use accuracy: if a model always predicted "Low stress" for every single person, it would still get 33% accuracy because the classes are balanced at about 1000 each. That model is completely useless but accuracy would not punish it.

F1 Macro calculates the F1 score separately for each class — Low, Moderate, High — and then averages them. This means the model is punished equally for doing badly on any one class. It cannot just focus on the easy class and ignore the others.

We also report Accuracy alongside F1 Macro since the classes are balanced — in this case both metrics tell a similar story.

---

## Q5. What model did you pick? Tradeoffs? Baseline vs improvement? Final metrics?

**Baseline: Decision Tree (max_depth=5)**

A single decision tree asks yes/no questions about the features and splits the data into branches. It is simple, fast, and completely visualisable — you can literally draw the decision logic as a flowchart. The problem is that one tree has high variance. A bad split at the top corrupts every prediction below it. It is also sensitive to which 80% of data it trained on.

Result: F1 Macro = 0.2819

**Improvement: XGBoost**

XGBoost builds 300 trees sequentially. Each new tree focuses specifically on the mistakes the previous trees made — this is called boosting. It handles complex feature interactions a single shallow tree misses. Regularisation through a low learning rate prevents overfitting.

After Optuna tuning: F1 Macro = 0.3669

The improvement of +0.086 F1 is confirmed by 5-fold cross-validation (0.3474 ± 0.017), which means the improvement holds across different data splits — it is not just a lucky test set.

---

## Q6. How would this model work in production? Batch scoring or streaming?

There are two deployment patterns depending on the use case:

**Real-time (what we built):** The FastAPI endpoint receives a person's lifestyle data and returns a stress prediction within milliseconds. This suits a mobile health app where the user fills in their daily habits and immediately sees their stress risk. We can scale this horizontally — run multiple API instances behind a load balancer for high traffic.

**Batch scoring:** Every night, run the model on all users in a database and flag anyone who moved from Moderate to High stress. Send them an alert or a recommendation. This suits an HR system monitoring employee wellness across a large organisation.

The model would be retrained monthly as new lifestyle data comes in, since habits and stress patterns change over time.

---

## Q7. What online metrics and data drift would you monitor?

**Prediction drift:** Monitor the distribution of predicted stress levels over time. If suddenly 70% of predictions are High stress when it was usually 33%, something has changed — either the user base changed or the model is no longer calibrated.

**Data drift:** Monitor the statistical distribution of incoming features. If the average sleep hours in new data drops from 6.5 to 4 hours, the model was trained on different patterns and predictions may be unreliable. Tools like Evidently AI can detect this automatically.

**Model performance decay:** If we have ground truth — follow-up surveys where users report their actual stress level — we can track real F1 over time. If it drops below a threshold, we trigger retraining.

**Latency:** Monitor API response time. If the model becomes a bottleneck under high load, we scale or optimise.

---

## Q8. Live demo — what to say

Start the API before the presentation:
```
uvicorn serving.main:app --reload
```

Open browser: http://localhost:8000/docs

Click POST /predict → Try it out → paste this request:

```json
{
  "age": 30,
  "sleep_hours": 4.0,
  "work_hours_per_week": 60,
  "screen_time_hours": 7.5,
  "social_interaction_score": 3.0,
  "happiness_score": 3.0,
  "exercise_level": "Low",
  "gender": "Male",
  "diet_type": "Junk Food",
  "country": "USA",
  "mental_health_condition": "None"
}
```

Click Execute. Expected response: **High stress**

**What to say:**
"Here is a real request through our FastAPI endpoint. This person sleeps 4 hours, works 60 hours a week, has no exercise, and high screen time. The model returns High stress with the highest probability. Notice we also get probabilities for all three classes — this is important because it tells us how confident the model is. The SHAP analysis in the notebook explains that sleep_debt and work_intensity are the two biggest drivers of this prediction."

---

## Common Questions the Teacher May Ask

**"Why is your F1 only 0.37?"**
Lifestyle features alone are a weak signal for stress. In the real world, stress is influenced by personal history, genetics, relationships, and many factors not captured here. 0.37 is above random (0.33 for 3 balanced classes) and the SHAP values confirm the model learns real patterns — it is not noise. With richer data — heart rate, cortisol levels, sleep quality sensors — performance would be much higher.

**"Why did you switch from Mental Health Condition?"**
This is actually a finding, not a failure. The EDA revealed that lifestyle habits alone cannot distinguish between Anxiety, Depression, Bipolar, and PTSD in this dataset. Recognising that and making a principled decision to switch targets is exactly what a real ML team does. Blindly submitting a 20% F1 model would be worse.

**"Why not use a neural network?"**
For tabular data of this size (3,000 rows, 14 features), XGBoost consistently outperforms neural networks. Neural networks need much more data to learn meaningful patterns on tabular data. XGBoost also gives us SHAP explainability for free — a neural network would require additional approximation methods.

**"How did you choose Optuna over GridSearch?"**
GridSearch exhaustively tests every combination in a predefined grid — with 3 parameters and 5 values each that is 125 trials. Optuna uses Bayesian optimisation — it learns from previous trials which regions of the search space are promising and focuses there. It finds better results in fewer trials, and it handles continuous parameter ranges like learning_rate naturally.

**"What does DVC do?"**
DVC is like Git but for large files — datasets and model artifacts. Git cannot store a 2MB model file or a CSV with 3000 rows efficiently. DVC stores these in Google Drive and commits only a small pointer file to git. This means anyone who clones the repo can run `dvc pull` and get the exact same data and model we used.
