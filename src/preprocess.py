import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sleep_debt'] = np.maximum(0, 8 - df['Sleep Hours'])
    df['screen_overload'] = (df['Screen Time per Day (Hours)'] > 6).astype(int)
    df['work_intensity'] = df['Work Hours per Week'] / 40
    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    oe = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
    df['exercise_enc'] = oe.fit_transform(df[['Exercise Level']])
    df['stress_enc'] = oe.fit_transform(df[['Stress Level']])

    le_gender = LabelEncoder()
    le_diet = LabelEncoder()
    le_country = LabelEncoder()
    le_target = LabelEncoder()

    df['gender_enc'] = le_gender.fit_transform(df['Gender'])
    df['diet_enc'] = le_diet.fit_transform(df['Diet Type'])
    df['country_enc'] = le_country.fit_transform(df['Country'])
    df['target'] = le_target.fit_transform(df['Mental Health Condition'])

    encoders = {
        'le_gender': le_gender,
        'le_diet': le_diet,
        'le_country': le_country,
        'le_target': le_target,
    }
    return df, encoders


FEATURES = [
    'Age', 'Sleep Hours', 'Work Hours per Week', 'Screen Time per Day (Hours)',
    'Social Interaction Score', 'Happiness Score',
    'sleep_debt', 'screen_overload', 'work_intensity',
    'exercise_enc', 'stress_enc', 'gender_enc', 'diet_enc', 'country_enc'
]


def get_X_y(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['target']
    return X, y
