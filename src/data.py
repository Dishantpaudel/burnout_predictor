import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["raw_path"])
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 595 NaN in Mental Health Condition = healthy people
    df["Mental Health Condition"] = df["Mental Health Condition"].fillna("None")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df
