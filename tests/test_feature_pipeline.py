import pandas as pd
from src.features.feature_pipeline import FeaturePipeline


pipeline = FeaturePipeline()

sample_data = pd.DataFrame([
    {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": None,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }
])

processed = pipeline.preprocess(sample_data)

print(processed)
