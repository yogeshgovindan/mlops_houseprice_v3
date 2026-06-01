from src.model_registry.versioning import (
    save_versioned_model
)
from src.features.feature_pipeline import (
    FeaturePipeline
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow.sklearn
import mlflow
import joblib
import os


class TrainingPipeline:

    def __init__(self):

        self.data_path = "data/housing.csv"

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # MLflow
        mlflow.set_tracking_uri(
            "sqlite:///mlflow.db"
        )

        mlflow.set_experiment(
            "house-price-v3"
        )

    def train(self):

        print(
            "Loading dataset..."
        )

        # ------------------------
        # LOAD DATA
        # ------------------------
        df = pd.read_csv(
            self.data_path
        )

        # ------------------------
        # FEATURE ENGINEERING
        # ------------------------
        print(
            "Applying feature engineering..."
        )

        pipeline = FeaturePipeline()

        df = pipeline.preprocess(
            df
        )

        # ------------------------
        # SPLIT FEATURES/TARGET
        # ------------------------
        X = df.drop(
            "median_house_value",
            axis=1
        )

        y = df[
            "median_house_value"
        ]

        # ------------------------
        # TRAIN TEST SPLIT
        # ------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ------------------------
        # MODEL
        # ------------------------
        print(
            "Training model..."
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        # ------------------------
        # PREDICTIONS
        # ------------------------
        preds = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        rmse = (
            mean_squared_error(
                y_test,
                preds
            ) ** 0.5
        )

        r2 = r2_score(
            y_test,
            preds
        )

        # ------------------------
        # MLFLOW LOGGING
        # ------------------------
        with mlflow.start_run():

            mlflow.log_metric(
                "MAE",
                mae
            )

            mlflow.log_metric(
                "RMSE",
                rmse
            )

            mlflow.log_metric(
                "R2",
                r2
            )

            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

        # ------------------------
        # SAVE VERSIONED MODEL
        # ------------------------
        versioned_path = (
            save_versioned_model(
                model
            )
        )

        print(
            f"Versioned model saved: {versioned_path}"
        )

        # ------------------------
        # SAVE LATEST MODEL
        # API USES THIS
        # ------------------------
        latest_model_path = (
            "artifacts/model.pkl"
        )

        joblib.dump(
            model,
            latest_model_path
        )

        print(
            "Latest model updated!"
        )

        # ------------------------
        # METRICS
        # ------------------------
        print(
            "\nTraining Complete"
        )

        print(
            "R2 Score:",
            round(r2, 4)
        )

        print(
            "RMSE:",
            round(rmse, 2)
        )

        print(
            "MAE:",
            round(mae, 2)
        )

        return model


if __name__ == "__main__":

    trainer = (
        TrainingPipeline()
    )

    trainer.train()
