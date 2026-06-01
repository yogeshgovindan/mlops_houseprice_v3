
import streamlit as st
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="MLOps Dashboard",
    layout="wide"
)

st.title(
    "🏠 House Price MLOps Dashboard"
)

st.write(
    "Real-time model monitoring"
)

# -------------------------
# LOAD LOGS
# -------------------------
log_file = (
    "logs/predictions.json"
)

logs = []

if os.path.exists(
    log_file
):

    with open(
        log_file,
        "r"
    ) as f:

        for line in f:

            logs.append(
                json.loads(line)
            )

# -------------------------
# MODEL VERSION
# -------------------------
models = glob.glob(
    "artifacts/model_v*.pkl"
)

active_model = (
    len(models)
)

# -------------------------
# STATS
# -------------------------
total_predictions = (
    len(logs)
)

latest_prediction = (
    logs[-1]["prediction"]
    if logs
    else "N/A"
)

# -------------------------
# METRIC CARDS
# -------------------------
col1, col2, col3 = (
    st.columns(3)
)

with col1:

    st.metric(
        "Total Predictions",
        total_predictions
    )

with col2:

    st.metric(
        "Latest Prediction",
        latest_prediction
    )

with col3:

    st.metric(
        "Active Model",
        f"v{active_model}"
    )

# -------------------------
# DRIFT STATUS
# -------------------------
st.subheader(
    "🚨 Drift Monitoring"
)

if logs:

    latest_log = logs[-1]

    if (
        "drift_report"
        in latest_log
    ):

        drift_report = (
            latest_log[
                "drift_report"
            ]
        )

        drift_found = any(

            feature.get(
                "drift_detected",
                False
            )

            for feature in (
                drift_report.values()
            )

            if isinstance(
                feature,
                dict
            )
        )

        if drift_found:

            st.error(
                "⚠ Drift Detected"
            )

        else:

            st.success(
                "✅ Model Stable"
            )

# -------------------------
# PREDICTION LOGS
# -------------------------
st.subheader(
    "📝 Prediction Logs"
)

if logs:

    df = pd.DataFrame(
        logs
    )

    st.dataframe(
        df,
        use_container_width=True
    )

else:

    st.warning(
        "No prediction logs found."
    )


# -------------------------
# VISUAL MONITORING
# -------------------------
st.subheader(
    "📈 Prediction Analytics"
)

if logs:

    df = pd.DataFrame(
        logs
    )

    if "prediction" in df.columns:

        # -------------------------
        # Trend Chart
        # -------------------------
        st.write(
            "### Prediction Trend"
        )

        fig, ax = plt.subplots()

        ax.plot(
            df["prediction"]
        )

        ax.set_xlabel(
            "Prediction Number"
        )

        ax.set_ylabel(
            "House Price"
        )

        st.pyplot(fig)

        # -------------------------
        # Distribution
        # -------------------------
        st.write(
            "### Prediction Distribution"
        )

        fig2, ax2 = plt.subplots()

        ax2.hist(
            df["prediction"]
        )

        ax2.set_xlabel(
            "House Price"
        )

        ax2.set_ylabel(
            "Frequency"
        )

        st.pyplot(fig2)


# -------------------------
# RETRAINING HISTORY
# -------------------------
st.subheader(
    "🔁 Retraining History"
)

retrain_file = (
    "logs/retraining_logs.json"
)

retrain_logs = []

if os.path.exists(
    retrain_file
):

    with open(
        retrain_file,
        "r"
    ) as f:

        for line in f:

            retrain_logs.append(
                json.loads(line)
            )

if retrain_logs:

    retrain_df = (
        pd.DataFrame(
            retrain_logs
        )
    )

    st.dataframe(
        retrain_df,
        use_container_width=True
    )

else:

    st.info(
        "No retraining history found."
    )
