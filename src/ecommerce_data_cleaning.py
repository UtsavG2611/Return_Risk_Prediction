from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "ecommerce_returns_synthetic_data.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "ecommerce_clean.csv"


def load_clean_ecommerce_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip()
    if "Return_Status" in df.columns:
        df = df.dropna(subset=["Return_Status"])
        df["Return_Status"] = df["Return_Status"].astype(str).str.strip().str.lower()
        mapping = {
            "returned": 1,
            "return approved": 1,
            "not returned": 0,
            "return rejected": 0,
        }
        df["returned"] = df["Return_Status"].map(mapping)
        df = df.dropna(subset=["returned"])
        df["returned"] = df["returned"].astype(int)
    id_cols = [c for c in ["Order_ID", "Product_ID", "User_ID"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    if "Return_Date" in df.columns:
        df["Return_Date"] = pd.to_datetime(df["Return_Date"], errors="coerce")
    if "Days_to_Return" not in df.columns and "Order_Date" in df.columns and "Return_Date" in df.columns:
        df["Days_to_Return"] = (df["Return_Date"] - df["Order_Date"]).dt.days
    if "Order_Date" in df.columns:
        df["order_weekday"] = df["Order_Date"].dt.weekday
    numeric_cols = ["Days_to_Return", "Product_Price", "Order_Quantity", "Discount_Applied", "User_Age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Days_to_Return" in df.columns:
        df["Days_to_Return"] = df["Days_to_Return"].fillna(0)
    df["Discount_Applied"] = df.get("Discount_Applied", 0).fillna(0)
    df["User_Age"] = df["User_Age"].fillna(df["User_Age"].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")
    df["discount_pct"] = df["Discount_Applied"] / (df["Product_Price"].replace(0, np.finfo(float).eps))
    df["order_value"] = df["Product_Price"] * df["Order_Quantity"]
    df["is_high_price"] = (df["Product_Price"] >= df["Product_Price"].median()).astype(int)
    df["is_long_return"] = (df["Days_to_Return"] > df["Days_to_Return"].median()).astype(int)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in ["order_date", "return_date"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return df
