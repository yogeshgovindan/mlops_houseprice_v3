import pandas as pd


def detect_drift(
    training_df,
    prediction_logs,
    threshold=0.30
):

    # enough predictions?
    if len(prediction_logs) < 5:

        return {
            "message":
            "Not enough predictions for drift detection"
        }

    # convert logs → dataframe
    incoming_df = pd.DataFrame([
        log["input"]
        for log in prediction_logs
    ])

    drift_report = {}

    # numeric columns only
    numeric_cols = incoming_df.select_dtypes(
        include=["number"]
    ).columns

    for col in numeric_cols:

        # skip missing columns
        if col not in training_df.columns:
            continue

        train_mean = (
            training_df[col]
            .mean()
        )

        incoming_mean = (
            incoming_df[col]
            .mean()
        )

        difference = abs(
            train_mean
            - incoming_mean
        )

        relative_shift = (
            difference
            / abs(train_mean)
            if train_mean != 0
            else 0
        )

        drift_report[col] = {

            "drift_detected":
            bool(
                relative_shift
                > threshold
            ),

            "shift":
            round(
                float(relative_shift),
                3
            )
        }

    return drift_report
