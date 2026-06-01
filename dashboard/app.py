import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://mlops-houseprice-api.azurewebsites.net/predict"

st.set_page_config(
    page_title="MLOps House Price Dashboard",
    layout="wide"
)

st.title("🏠 MLOps House Price Prediction Platform")

# -----------------------------
# SIDEBAR - SYSTEM STATUS
# -----------------------------
st.sidebar.header("⚙️ System Status")

if st.sidebar.button("Check API Health"):
    try:
        r = requests.get(API_URL.replace("/predict", "/health"))
        st.sidebar.success(r.json())
    except:
        st.sidebar.error("API Down")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.header("📊 Input Features")

col1, col2, col3 = st.columns(3)

with col1:
    longitude = st.number_input("Longitude", -124.0, -114.0, -122.23)
    latitude = st.number_input("Latitude", 32.0, 42.0, 37.88)
    housing_median_age = st.number_input("Median Age", 1, 100, 41)

with col2:
    total_rooms = st.number_input("Total Rooms", 1, 10000, 880)
    total_bedrooms = st.number_input("Total Bedrooms", 1, 5000, 129)
    population = st.number_input("Population", 1, 50000, 322)

with col3:
    households = st.number_input("Households", 1, 5000, 126)
    median_income = st.number_input("Median Income", 0.1, 20.0, 8.3)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
    )

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🚀 Predict House Price"):

    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        st.success("Prediction Successful 🎯")

        # -----------------------------
        # METRICS SECTION
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🏠 Predicted Price",
                f"${result['predicted_house_price']:,.2f}"
            )

        with col2:
            st.metric(
                "📅 Timestamp",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        # -----------------------------
        # DRIFT REPORT
        # -----------------------------
        st.subheader("📊 Drift Monitoring")

        drift_df = pd.DataFrame(result["drift_report"]).T
        drift_df.reset_index(inplace=True)
        drift_df.columns = ["Feature", "Drift Detected", "Shift"]

        st.dataframe(drift_df, use_container_width=True)

        # -----------------------------
        # DRIFT VISUALIZATION
        # -----------------------------
        fig = px.bar(
            drift_df,
            x="Feature",
            y="Shift",
            color="Drift Detected",
            title="Feature Drift Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # SAVE HISTORY
        # -----------------------------
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "time": datetime.now(),
            "price": result["predicted_house_price"]
        })

    except Exception as e:
        st.error(f"API Error: {e}")

# -----------------------------
# HISTORY SECTION
# -----------------------------
st.header("📈 Prediction History")

if "history" in st.session_state and len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    fig2 = px.line(
        history_df,
        x="time",
        y="price",
        title="Predicted Price Trend"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(history_df)

else:
    st.info("No predictions yet. Run a prediction to see history.")
