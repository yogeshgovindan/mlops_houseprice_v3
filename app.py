
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from src.features.feature_pipeline import (
    FeaturePipeline
)

from src.monitoring.logger import (
    log_prediction,
    load_recent_predictions
)

from src.monitoring.drift import (
    detect_drift
)

from src.monitoring.retraining import (
    check_retraining_need
)

from src.monitoring.auto_retrain import (
    trigger_retraining
)

from src.utils.data_loader import (
    get_training_data
)

# -------------------------
# APP
# -------------------------
app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load(
    "artifacts/model.pkl"
)

feature_pipeline = (
    FeaturePipeline()
)


# -------------------------
# INPUT SCHEMA
# -------------------------
class HouseInput(BaseModel):

    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():

    return {
        "status": "API running"
    }


# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/health")
def health():

    return {
        "model_loaded":
        model is not None
    }


# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
def predict(data: HouseInput):

    # -------------------------
    # INPUT DATAFRAME
    # -------------------------
    input_df = pd.DataFrame([
        data.model_dump()
    ])

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    processed_df = (
        feature_pipeline.preprocess(
            input_df
        )
    )

    # -------------------------
    # PREDICTION
    # -------------------------
    prediction = model.predict(
        processed_df
    )[0]

    prediction_value = round(
        float(prediction),
        2
    )

    # -------------------------
    # LOG PREDICTION
    # -------------------------
    log_prediction(
        input_data=data.model_dump(),
        prediction=prediction_value
    )

    # -------------------------
    # LOAD TRAINING DATA
    # -------------------------
    training_df = (
        get_training_data()
    )

    # -------------------------
    # LOAD RECENT PREDICTIONS
    # -------------------------
    recent_predictions = (
        load_recent_predictions()
    )

    # -------------------------
    # DRIFT DETECTION
    # -------------------------
    drift_report = (
        detect_drift(
            training_df,
            recent_predictions
        )
    )

    # -------------------------
    # RETRAIN CHECK
    # -------------------------
    retraining_status = (
        check_retraining_need(
            drift_report
        )
    )

    # -------------------------
    # AUTO RETRAIN
    # -------------------------
    auto_retrain_result = None

    if retraining_status.get(
        "retraining_needed"
    ):

        auto_retrain_result = (
            trigger_retraining()
        )

    # -------------------------
    # RESPONSE
    # -------------------------
    return {

        "predicted_house_price":
        prediction_value,

        "drift_report":
        drift_report,

        "retraining_status":
        retraining_status,

        "auto_retraining":
        auto_retrain_result
    }
