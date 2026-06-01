import pandas as pd


class FeaturePipeline:

    def __init__(self):
        pass

    def preprocess(self, df: pd.DataFrame):

        data = df.copy()

        # -----------------------
        # HANDLE MISSING VALUES
        # -----------------------
        data["total_bedrooms"] = data["total_bedrooms"].fillna(
            data["total_bedrooms"].median()
        )

        # -----------------------
        # HANDLE CATEGORICAL
        # -----------------------
        if "ocean_proximity" in data.columns:

            mapping = {
                "<1H OCEAN": 0,
                "INLAND": 1,
                "ISLAND": 2,
                "NEAR BAY": 3,
                "NEAR OCEAN": 4
            }

            data["ocean_proximity"] = (
                data["ocean_proximity"]
                .map(mapping)
            )

        return data
