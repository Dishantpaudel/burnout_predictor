import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


FEATURES = [
    "Age", "Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)",
    "Social Interaction Score", "Happiness Score",
    "sleep_debt", "screen_overload", "work_intensity",
    "exercise_enc", "gender_enc", "diet_enc", "country_enc", "cond_enc"
]

TARGET_NAMES = ["Low", "Moderate", "High"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived features — each motivated by EDA
    df["sleep_debt"]      = np.maximum(0, 8 - df["Sleep Hours"])
    df["screen_overload"] = (df["Screen Time per Day (Hours)"] > 6).astype(int)
    df["work_intensity"]  = df["Work Hours per Week"] / 40

    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    # Ordinal: preserve Low < Moderate < High order
    oe = OrdinalEncoder(categories=[["Low", "Moderate", "High"]])
    df["exercise_enc"] = oe.fit_transform(df[["Exercise Level"]])

    # Label encoders for nominal categoricals
    le_gender  = LabelEncoder()
    le_diet    = LabelEncoder()
    le_country = LabelEncoder()
    le_cond    = LabelEncoder()

    df["gender_enc"]  = le_gender.fit_transform(df["Gender"])
    df["diet_enc"]    = le_diet.fit_transform(df["Diet Type"])
    df["country_enc"] = le_country.fit_transform(df["Country"])
    df["cond_enc"]    = le_cond.fit_transform(df["Mental Health Condition"])

    # Ordinal encode target: Low=0, Moderate=1, High=2
    oe_target = OrdinalEncoder(categories=[["Low", "Moderate", "High"]])
    df["target"] = oe_target.fit_transform(df[["Stress Level"]]).astype(int)

    encoders = {
        "le_gender":  le_gender,
        "le_diet":    le_diet,
        "le_country": le_country,
        "le_cond":    le_cond,
        "oe_target":  oe_target,
    }
    return df, encoders


def get_X_y(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["target"]
    return X, y
