from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


st.title("Product Return Risk Prediction")


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "view" not in st.session_state:
    st.session_state["view"] = "predict"


def render_auth():
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.subheader("Login")
        login_userid = st.text_input("User ID", key="login_userid")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            payload = {"userid": login_userid, "password": login_password}
            try:
                response = requests.post("http://localhost:8000/login", json=payload, timeout=10)
            except Exception as e:
                st.error(f"Error calling login API: {e}")
            else:
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        st.session_state["logged_in"] = True
                        st.session_state["user_name"] = data.get("name") or ""
                    else:
                        st.error("Invalid user ID or password")
                else:
                    st.error(f"Login API returned status code {response.status_code}")

    with tab_register:
        st.subheader("Register")
        register_name = st.text_input("Name", key="register_name")
        register_userid = st.text_input("User ID", key="register_userid")
        register_password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register"):
            payload = {
                "name": register_name,
                "userid": register_userid,
                "password": register_password,
            }
            try:
                response = requests.post("http://localhost:8000/register", json=payload, timeout=10)
            except Exception as e:
                st.error(f"Error calling register API: {e}")
            else:
                if response.status_code == 200:
                    st.success("Registered successfully, please log in")
                else:
                    data = response.json()
                    detail = data.get("detail", "Registration failed")
                    st.error(detail)


def generate_plots() -> List[Tuple[str, str]]:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "raw" / "data.csv"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    plots: List[Tuple[str, str]] = []

    # 1. Return status count
    plt.figure(figsize=(6, 4))
    df["Return_Status"].value_counts().plot(kind="bar")
    plt.title("Return Status Count")
    plt.xlabel("Return Status")
    plt.ylabel("Count")
    path1 = plots_dir / "return_status_count.png"
    plt.tight_layout()
    plt.savefig(path1)
    plt.close()
    plots.append((str(path1), "Count of returned vs not returned orders"))

    # 2. Average price by category
    plt.figure(figsize=(8, 4))
    df.groupby("Product_Category")["Product_Price"].mean().sort_values().plot(kind="bar")
    plt.title("Average Product Price by Category")
    plt.xlabel("Product Category")
    plt.ylabel("Average Price")
    path2 = plots_dir / "avg_price_by_category.png"
    plt.tight_layout()
    plt.savefig(path2)
    plt.close()
    plots.append((str(path2), "Average product price for each category"))

    # 3. Return rate by category
    df_ret = df.copy()
    df_ret["returned_flag"] = df_ret["Return_Status"].str.strip().str.lower().eq("returned").astype(int)
    return_rate = df_ret.groupby("Product_Category")["returned_flag"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    return_rate.plot(kind="bar")
    plt.title("Return Rate by Category")
    plt.xlabel("Product Category")
    plt.ylabel("Return Rate")
    path3 = plots_dir / "return_rate_by_category.png"
    plt.tight_layout()
    plt.savefig(path3)
    plt.close()
    plots.append((str(path3), "Fraction of orders returned in each category"))

    # 4. Distribution of product price
    plt.figure(figsize=(6, 4))
    df["Product_Price"].plot(kind="hist", bins=20)
    plt.title("Distribution of Product Price")
    plt.xlabel("Product Price")
    plt.ylabel("Frequency")
    path4 = plots_dir / "price_distribution.png"
    plt.tight_layout()
    plt.savefig(path4)
    plt.close()
    plots.append((str(path4), "Histogram of product prices across orders"))

    # 5. Return rate by payment method
    return_rate_pay = df_ret.groupby("Payment_Method")["returned_flag"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    return_rate_pay.plot(kind="bar")
    plt.title("Return Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Return Rate")
    path5 = plots_dir / "return_rate_by_payment_method.png"
    plt.tight_layout()
    plt.savefig(path5)
    plt.close()
    plots.append((str(path5), "Return rate for each payment method"))

    # 6. Return rate by shipping method
    return_rate_ship = df_ret.groupby("Shipping_Method")["returned_flag"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    return_rate_ship.plot(kind="bar")
    plt.title("Return Rate by Shipping Method")
    plt.xlabel("Shipping Method")
    plt.ylabel("Return Rate")
    path6 = plots_dir / "return_rate_by_shipping_method.png"
    plt.tight_layout()
    plt.savefig(path6)
    plt.close()
    plots.append((str(path6), "Return rate for each shipping method"))

    # 7. Return rate by age band
    df_ret["Age_Band"] = pd.cut(
        df_ret["User_Age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
        include_lowest=True,
    )
    return_rate_age = df_ret.groupby("Age_Band")["returned_flag"].mean()
    plt.figure(figsize=(6, 4))
    return_rate_age.plot(kind="bar")
    plt.title("Return Rate by Age Band")
    plt.xlabel("Age Band")
    plt.ylabel("Return Rate")
    path7 = plots_dir / "return_rate_by_age_band.png"
    plt.tight_layout()
    plt.savefig(path7)
    plt.close()
    plots.append((str(path7), "Return rate for different customer age bands"))

    return plots


def render_prediction():
    DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "data.csv"

    category_options = []
    payment_options = []
    shipping_options = []
    try:
        df_cat = pd.read_csv(DATA_PATH)
        if "Product_Category" in df_cat.columns:
            category_options = sorted(df_cat["Product_Category"].dropna().unique().tolist())
        if "Payment_Method" in df_cat.columns:
            payment_options = sorted(df_cat["Payment_Method"].dropna().unique().tolist())
        if "Shipping_Method" in df_cat.columns:
            shipping_options = sorted(df_cat["Shipping_Method"].dropna().unique().tolist())
    except Exception:
        category_options = []
        payment_options = []
        shipping_options = []

    top_cols = st.columns([3, 1, 1])
    with top_cols[0]:
        if st.session_state["user_name"]:
            st.write(f"Welcome, {st.session_state['user_name']}")
    with top_cols[1]:
        if st.button("Dashboard"):
            st.session_state["view"] = "dashboard"
    with top_cols[2]:
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["user_name"] = ""
            st.session_state["view"] = "predict"

    product_category = (
        st.selectbox("Product Category", category_options)
        if category_options
        else st.text_input("Product Category")
    )
    product_price = st.number_input("Product Price", min_value=0.0, value=0.0)
    order_quantity = st.number_input("Order Quantity", min_value=1, value=1)
    user_age = st.number_input("User Age", min_value=0, max_value=120, value=0)
    user_gender = st.selectbox("User Gender", ["Male", "Female", "Other"])
    payment_method = (
        st.selectbox("Payment Method", payment_options)
        if payment_options
        else st.text_input("Payment Method")
    )
    shipping_method = (
        st.selectbox("Shipping Method", shipping_options)
        if shipping_options
        else st.text_input("Shipping Method")
    )
    discount_applied = st.number_input("Discount Applied", min_value=0.0, value=0.0)

    if st.button("Predict Return Risk"):
        payload = {
            "product_category": product_category,
            "product_price": float(product_price),
            "order_quantity": int(order_quantity),
            "user_age": int(user_age),
            "user_gender": user_gender,
            "payment_method": payment_method,
            "shipping_method": shipping_method,
            "discount_applied": float(discount_applied),
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        except Exception as e:
            st.error(f"Error calling prediction API: {e}")
        else:
            if response.status_code == 200:
                data = response.json()
                label = data.get("prediction_label", "No")
                if label == "Yes":
                    st.markdown(
                        "<span style='color:red; font-size:24px; font-weight:bold;'>Yes</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:green; font-size:24px; font-weight:bold;'>No</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.error(f"API returned status code {response.status_code}")


def render_dashboard():
    top_cols = st.columns([3, 1, 1])
    with top_cols[0]:
        st.subheader("Dashboard")
    with top_cols[1]:
        if st.button("Back to Prediction"):
            st.session_state["view"] = "predict"
    with top_cols[2]:
        if st.button("Logout", key="logout_dashboard"):
            st.session_state["logged_in"] = False
            st.session_state["user_name"] = ""
            st.session_state["view"] = "predict"

    plots = generate_plots()

    st.markdown("---")
    st.write("Overview of key patterns in returns, prices, and customer behavior.")

    for path, description in plots:
        col_img, col_text = st.columns([3, 2])
        with col_img:
            st.image(path, use_column_width=True)
        with col_text:
            st.write(description)
        st.markdown("---")


if not st.session_state["logged_in"]:
    render_auth()
else:
    if st.session_state["view"] == "dashboard":
        render_dashboard()
    else:
        render_prediction()
