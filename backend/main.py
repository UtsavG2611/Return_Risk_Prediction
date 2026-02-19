from pathlib import Path

import bcrypt
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import (
    LoginRequest,
    LoginResponse,
    OrderFeatures,
    PredictionResponse,
    RegisterRequest,
)
from database import create_user, get_user_by_userid, save_prediction


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

package = None
model = None
encoder = None

if MODEL_PATH.exists():
    package = joblib.load(MODEL_PATH)
    if isinstance(package, dict):
        model = package.get("model")
        encoder = package.get("encoder")
    else:
        model = package


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if "discount_applied" in df.columns and "product_price" in df.columns:
        denom = df["product_price"].replace(0, np.finfo(float).eps)
        df["discount_pct"] = df["discount_applied"] / denom
    if "product_price" in df.columns and "order_quantity" in df.columns:
        df["order_value"] = df["product_price"] * df["order_quantity"]
    return df


app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: OrderFeatures):
    if model is None:
        return PredictionResponse(prediction_label="No")
    data = pd.DataFrame([features.dict()])
    data_fe = feature_engineering(data)
    if encoder is not None:
        data_enc = encoder.transform(data_fe)
    else:
        data_enc = data_fe
    proba_array = model.predict_proba(data_enc)
    proba = float(proba_array[0][1])
    prediction = 1 if proba > 0.5 else 0
    label = "Yes" if prediction == 1 else "No"
    save_prediction(features.dict(), prediction, proba)
    return PredictionResponse(prediction_label=label)


@app.post("/register")
def register(request: RegisterRequest):
    password_bytes = request.password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    success = create_user(request.name, request.userid, hashed.decode("utf-8"))
    if not success:
        raise HTTPException(status_code=400, detail="User ID already exists")
    return {"message": "User registered successfully"}


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    user = get_user_by_userid(request.userid)
    if user is None:
        return LoginResponse(success=False, name=None)
    stored_hash = user["password_hash"].encode("utf-8")
    password_bytes = request.password.encode("utf-8")
    if not bcrypt.checkpw(password_bytes, stored_hash):
        return LoginResponse(success=False, name=None)
    return LoginResponse(success=True, name=user["name"])
