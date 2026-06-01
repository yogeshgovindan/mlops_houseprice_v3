
import subprocess
import json
import os
from datetime import datetime


RETRAIN_LOG = (
    "logs/retraining_logs.json"
)


def trigger_retraining():

    try:

        subprocess.run(

            [
                "python",
                "-m",
                "src.pipelines.training_pipeline"
            ],

            check=True
        )

        retrain_log = {

            "timestamp":
            str(
                datetime.now()
            ),

            "status":
            "success",

            "message":
            "Model retrained successfully."
        }

    except Exception as e:

        retrain_log = {

            "timestamp":
            str(
                datetime.now()
            ),

            "status":
            "failed",

            "message":
            str(e)
        }

    # save log
    os.makedirs(
        "logs",
        exist_ok=True
    )

    with open(
        RETRAIN_LOG,
        "a"
    ) as f:

        f.write(
            json.dumps(
                retrain_log
            )
            + "\n"
        )

    return retrain_log
