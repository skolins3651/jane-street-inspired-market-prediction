from pathlib import Path

import pandas as pd

from js_market_prediction.evaluation.utility import calculate_adapted_utility


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_and_splits_1d.parquet"
)

LOGISTIC_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_baseline_predictions.parquet"
)

UTILITY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_utility_scores.csv"
)

WEIGHT_COLUMNS = [
    "weight_equal",
    "weight_liquidity",
]

SPLIT_ORDER = [
    "train",
    "validation",
    "test",
]

BASELINE_ACTION_COLUMNS = {
    "always_pass": "action_always_pass",
    "always_take": "action_always_take",
    "momentum_20d_positive": "action_momentum_20d_positive",
    "logistic_regression": "action_logistic_regression",
}


def load_split_data(
    input_path: Path = SPLIT_DATA_PATH,
) -> pd.DataFrame:
    """Load the split-labeled feature dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Split data file not found: {input_path}"
        )

    return pd.read_parquet(input_path)


def load_logistic_predictions(
    input_path: Path = LOGISTIC_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Load saved logistic-regression baseline predictions."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Logistic prediction file not found: {input_path}"
        )

    return pd.read_parquet(input_path)


def validate_split_data(data: pd.DataFrame) -> None:
    """Validate the split-labeled feature dataset."""
    required_columns = {
        "date",
        "symbol",
        "split",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "weight_liquidity",
        "return_20d",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Split data is missing columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Split data is empty.")

    if "SPY" in set(data["symbol"]):
        raise ValueError("SPY should not appear in split data.")

    duplicate_rows = data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate date-symbol rows."
        )

    actual_splits = set(data["split"].unique())
    expected_splits = set(SPLIT_ORDER)

    if actual_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, "
            f"found {sorted(actual_splits)}."
        )

    if data[sorted(required_columns)].isna().sum().sum() > 0:
        raise ValueError(
            "Split data contains missing values in required columns."
        )


def validate_logistic_predictions(data: pd.DataFrame) -> None:
    """Validate saved logistic-regression predictions."""
    required_columns = {
        "date",
        "symbol",
        "action_logistic_regression",
        "probability_logistic_regression",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Logistic prediction data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Logistic prediction data is empty.")

    duplicate_rows = data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            "Logistic prediction data has duplicate date-symbol rows."
        )

    actions = set(data["action_logistic_regression"].unique())

    if not actions.issubset({0, 1}):
        raise ValueError(
            "action_logistic_regression must contain only 0 and 1."
        )


def add_baseline_actions(
    split_data: pd.DataFrame,
    logistic_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Add all baseline action columns to one scoring dataset."""
    scoring_data = split_data.copy()

    scoring_data["action_always_pass"] = 0
    scoring_data["action_always_take"] = 1

    scoring_data["action_momentum_20d_positive"] = (
        scoring_data["return_20d"] > 0
    ).astype("int64")

    logistic_actions = logistic_predictions[
        [
            "date",
            "symbol",
            "action_logistic_regression",
            "probability_logistic_regression",
        ]
    ].copy()

    scoring_data = scoring_data.merge(
        logistic_actions,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    if (
        scoring_data[
            [
                "action_logistic_regression",
                "probability_logistic_regression",
            ]
        ]
        .isna()
        .sum()
        .sum()
        > 0
    ):
        raise ValueError(
            "Missing logistic-regression predictions after merge."
        )

    scoring_data["action_logistic_regression"] = (
        scoring_data["action_logistic_regression"]
        .astype("int64")
    )

    return scoring_data


def score_baselines(scoring_data: pd.DataFrame) -> pd.DataFrame:
    """Score all baseline actions by split and weight column."""
    result_rows = []

    for split_name in SPLIT_ORDER:
        split_data = scoring_data.loc[
            scoring_data["split"] == split_name
        ].copy()

        for baseline_name, action_column in BASELINE_ACTION_COLUMNS.items():
            for weight_column in WEIGHT_COLUMNS:
                utility_metrics = calculate_adapted_utility(
                    data=split_data,
                    date_column="date",
                    response_column="resp_1d",
                    action_column=action_column,
                    weight_column=weight_column,
                )

                result_row = {
                    "split": split_name,
                    "baseline": baseline_name,
                    "weight_column": weight_column,
                    **utility_metrics,
                }

                result_rows.append(result_row)

    results = pd.DataFrame(result_rows)

    return results


def save_results(
    results: pd.DataFrame,
    output_path: Path = UTILITY_RESULTS_PATH,
) -> None:
    """Save baseline utility results to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Utility results file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Utility results file is empty: {output_path}"
        )


def print_results_summary(
    scoring_data: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """Print a compact utility-scoring summary."""
    print("Baseline utilities scored successfully.")
    print(f"Rows: {len(scoring_data):,}")
    print(f"First date: {scoring_data['date'].min().date()}")
    print(f"Last date: {scoring_data['date'].max().date()}")

    print()
    print("Validation and test utility summary:")
    display_columns = [
        "split",
        "baseline",
        "weight_column",
        "utility",
        "total_profit",
        "mean_daily_profit",
        "daily_profit_std",
        "c_stat",
        "action_rate",
        "mean_resp_taken",
        "num_days",
    ]

    display_results = results.loc[
        results["split"].isin(["validation", "test"]),
        display_columns,
    ].copy()

    print(
        display_results
        .sort_values(
            ["split", "baseline", "weight_column"]
        )
        .to_string(index=False)
    )

    print()
    print("Results saved successfully.")
    print(f"Output path: {UTILITY_RESULTS_PATH}")


def main() -> None:
    """Score baseline strategies with adapted utility."""
    split_data = load_split_data()
    logistic_predictions = load_logistic_predictions()

    validate_split_data(split_data)
    validate_logistic_predictions(logistic_predictions)

    scoring_data = add_baseline_actions(
        split_data=split_data,
        logistic_predictions=logistic_predictions,
    )

    results = score_baselines(scoring_data)

    print_results_summary(
        scoring_data=scoring_data,
        results=results,
    )

    save_results(results)


if __name__ == "__main__":
    main()