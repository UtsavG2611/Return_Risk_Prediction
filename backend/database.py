import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path(__file__).resolve().parent / "returns.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "product_category TEXT,"
        "product_price REAL,"
        "order_quantity INTEGER,"
        "user_age INTEGER,"
        "user_gender TEXT,"
        "payment_method TEXT,"
        "shipping_method TEXT,"
        "discount_applied REAL,"
        "prediction INTEGER,"
        "probability REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "name TEXT NOT NULL,"
        "userid TEXT NOT NULL UNIQUE,"
        "password_hash TEXT NOT NULL)"
    )
    return conn


def save_prediction(features: Dict[str, Any], prediction: int, probability: float):
    conn = get_connection()
    values = (
        features["product_category"],
        float(features["product_price"]),
        int(features["order_quantity"]),
        int(features["user_age"]),
        features["user_gender"],
        features["payment_method"],
        features["shipping_method"],
        float(features["discount_applied"]),
        int(prediction),
        float(probability),
    )
    conn.execute(
        "INSERT INTO predictions (product_category, product_price, "
        "order_quantity, user_age, user_gender, payment_method, "
        "shipping_method, discount_applied, prediction, probability) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()
    conn.close()


def create_user(name: str, userid: str, password_hash: str) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (name, userid, password_hash) VALUES (?, ?, ?)",
            (name, userid, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_userid(userid: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute(
        "SELECT id, name, userid, password_hash FROM users WHERE userid = ?",
        (userid,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "userid": row[2],
        "password_hash": row[3],
    }
