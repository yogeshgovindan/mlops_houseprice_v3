import os
import json
from datetime import datetime


LOG_FILE = "logs/predictions.json"


def log_prediction(
    input_data,
    prediction
):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    log_entry = {

        "timestamp":
        str(datetime.now()),

        "input":
        input_data,

        "prediction":
        prediction
    }

    with open(
        LOG_FILE,
        "a"
    ) as f:

        f.write(
            json.dumps(log_entry)
            + "\n"
        )


def load_recent_predictions(
    n=100
):

    if not os.path.exists(
        LOG_FILE
    ):
        return []

    with open(
        LOG_FILE,
        "r"
    ) as f:

        lines = f.readlines()

    recent = lines[-n:]

    return [
        json.loads(line)
        for line in recent
    ]
