import os
import pandas as pd


class DataIngestion:

    def __init__(self):

        self.dataset_url = (
            "https://raw.githubusercontent.com/"
            "ageron/handson-ml/master/"
            "datasets/housing/housing.csv"
        )

        self.output_path = "data/housing.csv"

    def download_data(self):

        print("Downloading dataset...")

        df = pd.read_csv(self.dataset_url)

        os.makedirs("data", exist_ok=True)

        df.to_csv(self.output_path, index=False)

        print("Dataset saved successfully!")
        print("Shape:", df.shape)

        return df


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.download_data()