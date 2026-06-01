import pandas as pd
from src.features.feature_pipeline import (
    FeaturePipeline
)


def get_training_data():

    df = pd.read_csv(
        "data/housing.csv"
    )

    pipeline = FeaturePipeline()

    df = pipeline.preprocess(df)

    return df
