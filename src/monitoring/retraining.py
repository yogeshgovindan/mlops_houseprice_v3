def check_retraining_need(
    drift_report,
    threshold=3
):

    drift_count = 0

    for feature, info in (
        drift_report.items()
    ):

        if (
            isinstance(info, dict)
            and info.get(
                "drift_detected"
            )
        ):

            drift_count += 1

    if drift_count >= threshold:

        return {
            "retraining_needed":
            True,

            "message":
            "High drift detected. Retraining recommended."
        }

    return {
        "retraining_needed":
        False,

        "message":
        "Model stable."
    }
