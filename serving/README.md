# Serving — FastAPI Stress Level Predictor

## Start the API

```bash
# from the burnout_predictor/ root folder
uvicorn serving.main:app --reload
```

## Open Swagger UI

```
http://localhost:8000/docs
```

Click **POST /predict → Try it out → Execute** to send a live request.

## Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/health` | Check if API is running |
| POST | `/predict` | Predict stress level from lifestyle input |

## Example Request

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

## Example Response

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

## Valid Input Values

| Field | Valid values |
|---|---|
| exercise_level | `Low`, `Moderate`, `High` |
| gender | `Male`, `Female`, `Other` |
| diet_type | `Balanced`, `Vegan`, `Vegetarian`, `Keto`, `Junk Food` |
| country | `USA`, `UK`, `India`, `Brazil`, `Australia`, `Germany`, `Japan`, `Canada` |
| mental_health_condition | `None`, `Anxiety`, `Depression`, `Bipolar`, `PTSD` |
