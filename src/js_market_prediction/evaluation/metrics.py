import numpy as np
import pandas as pd


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Divide two values, returning NaN when the denominator is zero."""
    if denominator == 0:
        return np.nan

    return numerator / denominator


def evaluate_actions(
    data: pd.DataFrame,
    action_column: str,
    weight_column: str,
) -> dict[str, float | int | str]:
    """Evaluate binary actions against target and response columns."""
    required_columns = {
        action_column,
        weight_column,
        "target_1d",
        "resp_1d",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    actions = data[action_column].astype("int64")
    targets = data["target_1d"].astype("int64")
    responses = data["resp_1d"]
    weights = data[weight_column]

    if not set(actions.unique()).issubset({0, 1}):
        raise ValueError(
            f"{action_column} must contain only 0 and 1."
        )

    row_count = len(data)
    trading_date_count = data["date"].nunique()
    symbol_count = data["symbol"].nunique()

    predicted_positive = actions == 1
    actual_positive = targets == 1
    correct = actions == targets

    true_positives = (
        predicted_positive
        & actual_positive
    ).sum()

    predicted_positives = predicted_positive.sum()
    actual_positives = actual_positive.sum()

    taken_responses = responses[predicted_positive]
    taken_weights = weights[predicted_positive]
    taken_targets = targets[predicted_positive]

    weighted_taken_denominator = taken_weights.sum()

    total_weighted_resp_taken = (
        taken_weights
        * taken_responses
    ).sum()

    weighted_true_positives_taken = (
        taken_weights
        * taken_targets
    ).sum()

    metrics = {
        "row_count": row_count,
        "trading_date_count": trading_date_count,
        "symbol_count": symbol_count,
        "target_rate": targets.mean(),
        "accuracy": correct.mean(),
        "precision": safe_divide(
            true_positives,
            predicted_positives,
        ),
        "recall": safe_divide(
            true_positives,
            actual_positives,
        ),
        "predicted_positive_rate": actions.mean(),
        "mean_resp_taken": (
            taken_responses.mean()
            if predicted_positives > 0
            else np.nan
        ),
        "weighted_mean_resp_taken": safe_divide(
            total_weighted_resp_taken,
            weighted_taken_denominator,
        ),
        "total_weighted_resp_taken": total_weighted_resp_taken,
        "weighted_hit_rate_taken": safe_divide(
            weighted_true_positives_taken,
            weighted_taken_denominator,
        ),
    }

    return metrics