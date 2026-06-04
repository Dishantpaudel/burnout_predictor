import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

from preprocess import load_data, engineer_features, encode_features, get_X_y, FEATURES

DATA_PATH = '../data/raw/Mental_Health_Lifestyle_Dataset.csv'
MODEL_DIR = '../app/model'


def train():
    print("=== TRAINING PIPELINE ===\n")

    df = load_data(DATA_PATH)
    df = engineer_features(df)
    df, encoders = encode_features(df)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    le_target = encoders['le_target']
    print(f"Test Accuracy : {acc:.4f}")
    print(f"F1 Macro      : {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    cv = cross_val_score(model, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f"5-Fold CV F1  : {cv.mean():.4f} ± {cv.std():.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,                  f'{MODEL_DIR}/model.pkl')
    joblib.dump(le_target,              f'{MODEL_DIR}/label_encoder.pkl')
    joblib.dump(FEATURES,               f'{MODEL_DIR}/feature_names.pkl')
    joblib.dump({k: v for k, v in encoders.items() if k != 'le_target'},
                f'{MODEL_DIR}/encoders.pkl')
    print(f"\nModel saved to {MODEL_DIR}/")


if __name__ == '__main__':
    train()
