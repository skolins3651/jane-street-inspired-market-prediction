from pathlib import Path

import pandas as pd

from js_market_prediction.evaluation.metrics import evaluate_actions


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_and_splits_1d.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "non_ml_baseline_results.csv"
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


def load_split_data(
    input_path: Path = SPLIT_DATA_PATH,
) -> pd.DataFrame:
    """Load the split-labeled feature dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Split data file not found: {input_path}"
        )

    data = pd.read_parquet(input_path)

    return data


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
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Split dataset is empty.")

    expected_splits = set(SPLIT_ORDER)
    actual_splits = set(data["split"].unique())

    if actual_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, "
            f"found {sorted(actual_splits)}."
        )

    if data.isna().sum().sum() > 0:
        raise ValueError("Split dataset contains missing values.")

    if "SPY" in set(data["symbol"]):
        raise ValueError("SPY should not appear in the split dataset.")

    duplicate_rows = data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate date-symbol rows."
        )


def add_baseline_actions(data: pd.DataFrame) -> pd.DataFrame:
    """Add non-ML baseline action columns."""
    baseline_data = data.copy()

    baseline_data["action_always_pass"] = 0
    baseline_data["action_always_take"] = 1

    baseline_data["action_momentum_20d_positive"] = (
        baseline_data["return_20d"] > 0
    ).astype("int64")

    return baseline_data


def evaluate_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate non-ML baselines by split and weight column."""
    baseline_actions = {
        "always_pass": "action_always_pass",
        "always_take": "action_always_take",
        "momentum_20d_positive": "action_momentum_20d_positive",
    }

    result_rows = []

    for split_name in SPLIT_ORDER:
        split_data = data.loc[
            data["split"] == split_name
        ].copy()

        for baseline_name, action_column in baseline_actions.items():
            for weight_column in WEIGHT_COLUMNS:
                metrics = evaluate_actions(
                    data=split_data,
                    action_column=action_column,
                    weight_column=weight_column,
                )

                result_row = {
                    "split": split_name,
                    "baseline": baseline_name,
                    "weight_column": weight_column,
                    **metrics,
                }

                result_rows.append(result_row)

    results = pd.DataFrame(result_rows)

    return results


def save_results(
    results: pd.DataFrame,
    output_path: Path = RESULTS_PATH,
) -> None:
    """Save baseline results to CSV."""
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
            f"Results file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Results file is empty: {output_path}"
        )


def print_results_summary(
    data: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """Print a compact summary of non-ML baseline results."""
    print("Non-ML baselines evaluated successfully.")
    print(f"Input path: {SPLIT_DATA_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"First date: {data['date'].min().date()}")
    print(f"Last date: {data['date'].max().date()}")

    print()
    print("Results table:")
    display_columns = [
        "split",
        "baseline",
        "weight_column",
        "row_count",
        "target_rate",
        "accuracy",
        "precision",
        "recall",
        "predicted_positive_rate",
        "mean_resp_taken",
        "weighted_mean_resp_taken",
        "total_weighted_resp_taken",
        "weighted_hit_rate_taken",
    ]

    print(
        results[display_columns]
        .sort_values(
            ["split", "baseline", "weight_column"]
        )
        .to_string(index=False)
    )

    print()
    print("Results saved successfully.")
    print(f"Output path: {RESULTS_PATH}")


def main() -> None:
    """Run non-ML baseline strategies."""
    split_data = load_split_data()
    validate_split_data(split_data)

    baseline_data = add_baseline_actions(split_data)
    results = evaluate_baselines(baseline_data)

    print_results_summary(
        data=baseline_data,
        results=results,
    )

    save_results(results)


if __name__ == "__main__":
    main()