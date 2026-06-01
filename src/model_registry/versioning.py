import os
import joblib


MODEL_DIR = "artifacts"


def save_versioned_model(
    model
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    existing_models = [

        f for f in os.listdir(
            MODEL_DIR
        )

        if f.startswith(
            "model_v"
        )
    ]

    version = (
        len(existing_models)
        + 1
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"model_v{version}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    return model_path
