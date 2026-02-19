from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "Export_Product_Return_Data.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "clean_export.csv"


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    first_col = df.columns[0]
    df = df.drop(columns=[first_col])
    if "Brand" in df.columns:
        df = df.drop(columns=["Brand"])
    df = df.rename(
        columns={
            "Age": "age",
            "Gender": "gender",
            "State": "state",
            "Category": "category",
            "Quantity": "quantity",
            "Price": "price",
            "Discount": "discount",
            "Product Rating": "rating",
            "high_return_risk": "returned",
        }
    )
    df = df.dropna(subset=["returned"])
    df["returned"] = df["returned"].map({"Yes": 1, "No": 0})
    df.to_csv(PROCESSED_PATH, index=False)
    return df
