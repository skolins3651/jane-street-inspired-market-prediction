from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_adapted_utility(
    data: pd.DataFrame,
    date_column: str = "date",
    response_column: str = "resp_1d",
    action_column: str = "action",
    weight_column: str = "weight_liquidity",
) -> dict[str, float | int]:
    """Calculate the adapted Jane Street-style utility score.

    Daily profit is the sum of weighted selected responses by date:

        p_t = sum(weight * response * action)

    The final utility score is:

        utility = min(max(c_stat, 0), 6) * total_profit

    where:

        c_stat = total_profit / sqrt(sum(daily_profit^2)) * sqrt(250 / num_days)
    """
    required_columns = {
        date_column,
        response_column,
        action_column,
        weight_column,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Cannot score an empty dataset.")

    scoring_data = data[
        [
            date_column,
            response_column,
            action_column,
            weight_column,
        ]
    ].copy()

    if scoring_data.isna().sum().sum() > 0:
        raise ValueError(
            "Scoring data contains missing values in required columns."
        )

    actions = scoring_data[action_column].astype("int64")

    if not set(actions.unique()).issubset({0, 1}):
        raise ValueError(
            f"{action_column} must contain only 0 and 1."
        )

    weights = scoring_data[weight_column].astype("float64")

    if (weights < 0).any():
        raise ValueError(
            f"{weight_column} must contain nonnegative weights."
        )

    responses = scoring_data[response_column].astype("float64")

    if not np.isfinite(responses.to_numpy()).all():
        raise ValueError(
            f"{response_column} contains non-finite values."
        )

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError(
            f"{weight_column} contains non-finite values."
        )

    scoring_data["_selected_profit"] = (
        weights
        * responses
        * actions
    )

    daily_profit = (
        scoring_data
        .groupby(date_column)["_selected_profit"]
        .sum()
        .sort_index()
    )

    num_days = int(daily_profit.shape[0])
    total_profit = float(daily_profit.sum())
    sum_squared_daily_profit = float(
        np.square(daily_profit).sum()
    )

    if sum_squared_daily_profit == 0.0:
        c_stat = 0.0
        utility = 0.0
    else:
        c_stat = float(
            total_profit
            / np.sqrt(sum_squared_daily_profit)
            * np.sqrt(250 / num_days)
        )

        utility = float(
            min(
                max(c_stat, 0.0),
                6.0,
            )
            * total_profit
        )

    action_rate = float(actions.mean())

    taken_mask = actions == 1

    if taken_mask.sum() == 0:
        mean_resp_taken = np.nan
    else:
        mean_resp_taken = float(
            responses[taken_mask].mean()
        )

    results = {
        "utility": utility,
        "total_profit": total_profit,
        "mean_daily_profit": float(daily_profit.mean()),
        "daily_profit_std": float(daily_profit.std(ddof=0)),
        "c_stat": c_stat,
        "action_rate": action_rate,
        "mean_resp_taken": mean_resp_taken,
        "num_days": num_days,
    }

    return results