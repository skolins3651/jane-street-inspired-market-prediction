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

THRESHOLD_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_threshold_tuning_results.csv"
)

SELECTED_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_threshold_tuning_selected.csv"
)

THRESHOLDS = [
    0.45,
    0.46,
    0.47,
    0.48,
    0.49,
    0.50,
    0.51,
    0.52,
    0.53,
    0.54,
    0.55,
]

WEIGHT_COLUMNS = ["weight_equal", "weight_liquidity"]

MAIN_SELECTION_WEIGHT = "weight_liquidity"
TUNING_SPLIT = "validation"


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split dataset not found: {path}")

    data = pd.read_parquet(path)
    validate_split_data(data)
    return data


def load_logistic_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Logistic-regression predictions not found: {path}\n"
            "Run this first:\n"
            "python -m js_market_prediction.models.run_logistic_regression_baseline"
        )

    predictions = pd.read_parquet(path)
    validate_logistic_predictions(predictions)
    return predictions


def validate_split_data(data: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "symbol",
        "split",
        "target_1d",
        "resp_1d",
        "weight_equal",
        "weight_liquidity",
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Split dataset is missing columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Split dataset is empty.")

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Split dataset has missing values in required columns.")


def validate_logistic_predictions(predictions: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "symbol",
        "probability_logistic_regression",
    }

    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Logistic-regression predictions are missing columns: {sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError("Logistic-regression predictions are empty.")

    if predictions[list(required_columns)].isna().any().any():
        raise ValueError(
            "Logistic-regression predictions have missing values in required columns."
        )

    probabilities = predictions["probability_logistic_regression"]

    if not probabilities.between(0.0, 1.0).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")


def merge_predictions(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = [
        "date",
        "symbol",
        "probability_logistic_regression",
    ]

    merged = data.merge(
        predictions[prediction_columns],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    if merged["probability_logistic_regression"].isna().any():
        missing_count = merged["probability_logistic_regression"].isna().sum()
        raise ValueError(f"Missing logistic predictions after merge: {missing_count}")

    return merged


def calculate_classification_metrics(data: pd.DataFrame, action_column: str) -> dict[str, float]:
    actual = data["target_1d"].astype(int)
    action = data[action_column].astype(int)

    true_positive = ((actual == 1) & (action == 1)).sum()
    false_positive = ((actual == 0) & (action == 1)).sum()
    false_negative = ((actual == 1) & (action == 0)).sum()

    total_rows = len(data)
    predicted_positive = action.sum()
    actual_positive = actual.sum()

    accuracy = (actual == action).mean()
    action_rate = predicted_positive / total_rows

    if predicted_positive == 0:
        precision = float("nan")
    else:
        precision = true_positive / (true_positive + false_positive)

    if actual_positive == 0:
        recall = float("nan")
    else:
        recall = true_positive / (true_positive + false_negative)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "action_rate": action_rate,
    }


def score_thresholds(data: pd.DataFrame) -> pd.DataFrame:
    tuning_data = data[data["split"] == TUNING_SPLIT].copy()

    if tuning_data.empty:
        raise ValueError(f"No rows found for tuning split: {TUNING_SPLIT}")

    rows = []

    for threshold in THRESHOLDS:
        scoring_data = tuning_data.copy()
        scoring_data["action_threshold"] = (
            scoring_data["probability_logistic_regression"] >= threshold
        ).astype("int64")

        classification_metrics = calculate_classification_metrics(
            scoring_data,
            action_column="action_threshold",
        )

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                scoring_data,
                action_column="action_threshold",
                weight_column=weight_column,
            )

            rows.append(
                {
                    "split": TUNING_SPLIT,
                    "threshold": threshold,
                    "weight_column": weight_column,
                    **classification_metrics,
                    **utility_metrics,
                }
            )

    return pd.DataFrame(rows)


def select_best_threshold(results: pd.DataFrame) -> pd.DataFrame:
    selection_data = results[results["weight_column"] == MAIN_SELECTION_WEIGHT].copy()

    if selection_data.empty:
        raise ValueError(
            f"No threshold results found for main selection weight: {MAIN_SELECTION_WEIGHT}"
        )

    selection_data = selection_data.sort_values(
        by=["utility", "total_profit", "mean_resp_taken"],
        ascending=[False, False, False],
    )

    return selection_data.head(1)


def save_results(results: pd.DataFrame, selected_threshold: pd.DataFrame) -> None:
    THRESHOLD_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(THRESHOLD_RESULTS_PATH, index=False)
    selected_threshold.to_csv(SELECTED_THRESHOLD_PATH, index=False)


def print_results_summary(results: pd.DataFrame, selected_threshold: pd.DataFrame) -> None:
    summary_columns = [
        "split",
        "threshold",
        "weight_column",
        "utility",
        "total_profit",
        "c_stat",
        "action_rate",
        "mean_resp_taken",
    ]

    print("\nThreshold tuning results:")
    print(
        results[summary_columns]
        .sort_values(["weight_column", "threshold"])
        .to_string(index=False)
    )

    print("\nSelected threshold using validation liquidity-weighted utility:")
    print(selected_threshold[summary_columns].to_string(index=False))

    print(f"\nSaved threshold results to: {THRESHOLD_RESULTS_PATH}")
    print(f"Saved selected threshold to: {SELECTED_THRESHOLD_PATH}")


def main() -> None:
    split_data = load_split_data(SPLIT_DATA_PATH)
    logistic_predictions = load_logistic_predictions(LOGISTIC_PREDICTIONS_PATH)

    scoring_data = merge_predictions(split_data, logistic_predictions)
    results = score_thresholds(scoring_data)
    selected_threshold = select_best_threshold(results)

    save_results(results, selected_threshold)
    print_results_summary(results, selected_threshold)

    print("\nLogistic threshold tuning completed successfully.")


if __name__ == "__main__":
    main()