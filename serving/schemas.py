from pydantic import BaseModel, Field
from typing import Dict


class LifestyleInput(BaseModel):
    age: int = Field(..., ge=18, le=100, example=30)
    sleep_hours: float = Field(..., ge=0, le=24, example=5.5)
    work_hours_per_week: int = Field(..., ge=0, le=100, example=55)
    screen_time_hours: float = Field(..., ge=0, le=24, example=7.0)
    social_interaction_score: float = Field(..., ge=1, le=10, example=4.0)
    happiness_score: float = Field(..., ge=1, le=10, example=3.5)
    exercise_level: str = Field(..., example="Low")
    gender: str = Field(..., example="Male")
    diet_type: str = Field(..., example="Junk Food")
    country: str = Field(..., example="USA")
    mental_health_condition: str = Field(..., example="None")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    predicted_stress_level: str = Field(..., example="High")
    probabilities: Dict[str, float] = Field(
        ...,
        example={"Low": 0.12, "Moderate": 0.23, "High": 0.65}
    )
