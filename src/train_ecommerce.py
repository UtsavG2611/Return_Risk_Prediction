from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from category_encoders import TargetEncoder

from .ecommerce_data_cleaning import load_clean_ecommerce_data


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "ecommerce_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "ecommerce_threshold.txt"


def optimize_threshold(y_true, probs):
    thresholds = np.linspace(0.1, 0.9, 100)
    best_f1 = 0.0
    best_threshold = 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
    return best_threshold, best_f1


def train():
    df = load_clean_ecommerce_data()
    X = df.drop(columns=["returned"])
    y = df["returned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoder = TargetEncoder(cols=categorical_cols) if categorical_cols else None
    if encoder is not None:
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
    else:
        X_train_enc = X_train
        X_test_enc = X_test
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_enc, y_train)
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated_model.fit(X_train_enc, y_train)
    probs = calibrated_model.predict_proba(X_test_enc)[:, 1]
    best_threshold, best_f1 = optimize_threshold(y_test, probs)
    preds = (probs >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, preds)
    print(f"Optimized Threshold: {best_threshold:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Best F1: {best_f1:.3f}")
    final_package = {
        "model": calibrated_model,
        "encoder": encoder,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_package, MODEL_PATH)
    THRESHOLD_PATH.write_text(str(best_threshold))
    print("Ecommerce model and threshold saved successfully.")


if __name__ == "__main__":
    train()

