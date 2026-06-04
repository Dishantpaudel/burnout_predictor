"""
Main training pipeline.

What this does:
  1. Loads config.yaml
  2. Loads and cleans data
  3. Engineers features
  4. Splits into train / val / test
  5. Runs Optuna to find best hyperparameters (50 trials)
  6. Logs every trial to MLflow
  7. Trains final model on best params
  8. Saves model artifacts to app/model/

Run with:
    cd burnout_predictor
    python src/train.py
"""

import os
import sys
import joblib
import mlflow
import mlflow.xgboost
import optuna
import yaml
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score

# Add src/ to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from data import load_config, load_data, clean_data
from features import engineer_features, encode_features, get_X_y, FEATURES, TARGET_NAMES
from evaluate import print_metrics


# ── silence optuna logs ───────────────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_training(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    df = load_data(cfg)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_features(df)
    X, y = get_X_y(df)

    # ── 2. Split: train / val / test ──────────────────────────────────────────
    rs   = cfg["split"]["random_state"]
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=rs, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=cfg["split"]["val_size"], random_state=rs, stratify=y_tv
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── 3. MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── 4. Optuna objective ───────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
            "max_depth":     trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric":   "mlogloss",
            "random_state":  rs,
            "n_jobs":        -1,
            "verbosity":     0,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val), average="macro")

        # Log each trial as its own MLflow run
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_f1_macro", val_f1)

        return val_f1

    # ── 5. Run Optuna ─────────────────────────────────────────────────────────
    print(f"\nRunning Optuna ({cfg['optuna']['n_trials']} trials)...")
    with mlflow.start_run(run_name="optuna_search"):
        study = optuna.create_study(direction=cfg["optuna"]["direction"])
        study.optimize(objective, n_trials=cfg["optuna"]["n_trials"])

        best_params = study.best_params
        best_val_f1 = study.best_value
        print(f"Best val F1: {best_val_f1:.4f}")
        print(f"Best params: {best_params}")

        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_f1_macro", best_val_f1)

    # ── 6. Train final model on best params ───────────────────────────────────
    print("\nTraining final model on best params...")
    final_params = {
        **best_params,
        "eval_metric":  "mlogloss",
        "random_state": rs,
        "n_jobs":       -1,
        "verbosity":    0,
    }
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train)

    # ── 7. Evaluate on test set ───────────────────────────────────────────────
    y_pred = final_model.predict(X_test)
    metrics = print_metrics(y_test, y_pred, "Final XGBoost (best params)")

    cv = cross_val_score(final_model, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"5-Fold CV F1: {cv.mean():.4f} +/- {cv.std():.4f}")

    # ── 8. Log final run to MLflow ────────────────────────────────────────────
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(final_params)
        mlflow.log_metric("test_accuracy",    metrics["accuracy"])
        mlflow.log_metric("test_f1_macro",    metrics["f1_macro"])
        mlflow.log_metric("cv_f1_mean",       cv.mean())
        mlflow.log_metric("cv_f1_std",        cv.std())
        mlflow.xgboost.log_model(final_model, artifact_path="model")

    # ── 9. Save artifacts ─────────────────────────────────────────────────────
    model_dir = cfg["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(final_model,                    cfg["paths"]["model_file"])
    joblib.dump(encoders["oe_target"],          cfg["paths"]["encoder_file"])
    joblib.dump(FEATURES,                       cfg["paths"]["feature_names_file"])
    joblib.dump(TARGET_NAMES,                   f"{model_dir}/target_names.pkl")
    joblib.dump({k: v for k, v in encoders.items() if k != "oe_target"},
                cfg["paths"]["encoders_file"])

    print(f"\nArtifacts saved to {model_dir}/")
    print("Done. Run 'mlflow ui' to view experiment results.")


if __name__ == "__main__":
    run_training()
