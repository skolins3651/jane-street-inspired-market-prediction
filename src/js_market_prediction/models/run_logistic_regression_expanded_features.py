from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from js_market_prediction.evaluation.utility import calculate_adapted_utility
from js_market_prediction.features.build_expanded_features import (
    EXPANDED_FEATURE_COLUMNS,
    ORIGINAL_FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPANDED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_expanded_features_and_splits_1d.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_expanded_features_results.csv"
)

UTILITY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_expanded_features_utility_scores.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_expanded_features_predictions.parquet"
)

COEFFICIENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_expanded_features_coefficients.csv"
)

FEATURE_COLUMNS = ORIGINAL_FEATURE_COLUMNS + EXPANDED_FEATURE_COLUMNS

SPLIT_ORDER = ["train", "validation", "test"]
WEIGHT_COLUMNS = ["weight_equal", "weight_liquidity"]

ACTION_THRESHOLD = 0.50


def load_expanded_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expanded feature dataset not found: {path}\n"
            "Run this first:\n"
            "python -m js_market_prediction.features.build_expanded_features"
        )

    data = pd.read_parquet(path)
    validate_expanded_data(data)
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def validate_expanded_data(data: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "symbol",
        "split",
        "target_1d",
        "resp_1d",
        "weight_equal",
        "weight_liquidity",
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Expanded feature dataset is missing columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Expanded feature dataset is empty.")

    if data[["date", "symbol"]].duplicated().any():
        raise ValueError("Expanded feature dataset has duplicate date-symbol rows.")

    required_data = data[list(required_columns)]

    if required_data.isna().any().any():
        raise ValueError("Expanded feature dataset has missing values in required columns.")

    if not np.isfinite(data[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Expanded feature dataset has non-finite feature values.")

    expected_splits = set(SPLIT_ORDER)
    observed_splits = set(data["split"].unique())

    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, found {sorted(observed_splits)}"
        )


def train_model(data: pd.DataFrame) -> Pipeline:
    train_data = data[data["split"] == "train"].copy()

    if train_data.empty:
        raise ValueError("Training split is empty.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(train_data[FEATURE_COLUMNS], train_data["target_1d"].astype(int))

    return model


def add_predictions(data: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    data = data.copy()

    probabilities = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]

    data["probability_expanded_logistic_regression"] = probabilities
    data["action_expanded_logistic_regression"] = (
        data["probability_expanded_logistic_regression"] >= ACTION_THRESHOLD
    ).astype("int64")

    return data


def calculate_classification_metrics(data: pd.DataFrame) -> dict[str, float]:
    actual = data["target_1d"].astype(int)
    action = data["action_expanded_logistic_regression"].astype(int)

    true_positive = ((actual == 1) & (action == 1)).sum()
    false_positive = ((actual == 0) & (action == 1)).sum()
    false_negative = ((actual == 1) & (action == 0)).sum()

    predicted_positive = action.sum()
    actual_positive = actual.sum()

    accuracy = (actual == action).mean()
    action_rate = action.mean()

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


def evaluate_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_rows = []
    utility_rows = []

    for split in SPLIT_ORDER:
        split_data = data[data["split"] == split].copy()

        if split_data.empty:
            raise ValueError(f"No rows found for split: {split}")

        classification_metrics = calculate_classification_metrics(split_data)

        classification_rows.append(
            {
                "model": "logistic_regression_expanded_features",
                "split": split,
                "threshold": ACTION_THRESHOLD,
                **classification_metrics,
            }
        )

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                split_data,
                action_column="action_expanded_logistic_regression",
                weight_column=weight_column,
            )

            utility_rows.append(
                {
                    "model": "logistic_regression_expanded_features",
                    "split": split,
                    "threshold": ACTION_THRESHOLD,
                    "weight_column": weight_column,
                    **classification_metrics,
                    **utility_metrics,
                }
            )

    return pd.DataFrame(classification_rows), pd.DataFrame(utility_rows)


def build_coefficient_table(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    return coefficient_table.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def save_outputs(
    prediction_data: pd.DataFrame,
    classification_results: pd.DataFrame,
    utility_results: pd.DataFrame,
    coefficient_table: pd.DataFrame,
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    classification_results.to_csv(RESULTS_PATH, index=False)
    utility_results.to_csv(UTILITY_RESULTS_PATH, index=False)

    prediction_columns = [
        "date",
        "symbol",
        "split",
        "target_1d",
        "resp_1d",
        "weight_equal",
        "weight_liquidity",
        "probability_expanded_logistic_regression",
        "action_expanded_logistic_regression",
    ]

    prediction_data[prediction_columns].to_parquet(PREDICTIONS_PATH, index=False)
    coefficient_table.to_csv(COEFFICIENTS_PATH, index=False)


def print_summary(
    prediction_data: pd.DataFrame,
    classification_results: pd.DataFrame,
    utility_results: pd.DataFrame,
    coefficient_table: pd.DataFrame,
) -> None:
    print("\nExpanded-feature logistic-regression prediction summary:")
    prediction_summary = prediction_data.groupby("split").agg(
        row_count=("target_1d", "size"),
        target_rate=("target_1d", "mean"),
        average_probability=("probability_expanded_logistic_regression", "mean"),
        action_rate=("action_expanded_logistic_regression", "mean"),
    )
    print(prediction_summary.loc[SPLIT_ORDER].to_string())

    print("\nClassification results:")
    print(classification_results.to_string(index=False))

    summary_columns = [
        "model",
        "split",
        "weight_column",
        "utility",
        "total_profit",
        "c_stat",
        "action_rate",
        "mean_resp_taken",
        "num_days",
    ]

    print("\nUtility results:")
    print(
        utility_results[summary_columns]
        .sort_values(["split", "weight_column"])
        .to_string(index=False)
    )

    print("\nTop coefficients by absolute value:")
    print(coefficient_table.head(15).to_string(index=False))

    print(f"\nSaved classification results to: {RESULTS_PATH}")
    print(f"Saved utility results to:        {UTILITY_RESULTS_PATH}")
    print(f"Saved predictions to:            {PREDICTIONS_PATH}")
    print(f"Saved coefficients to:           {COEFFICIENTS_PATH}")


def main() -> None:
    data = load_expanded_data(EXPANDED_DATA_PATH)

    model = train_model(data)
    prediction_data = add_predictions(data, model)

    classification_results, utility_results = evaluate_model(prediction_data)
    coefficient_table = build_coefficient_table(model)

    save_outputs(
        prediction_data=prediction_data,
        classification_results=classification_results,
        utility_results=utility_results,
        coefficient_table=coefficient_table,
    )

    print_summary(
        prediction_data=prediction_data,
        classification_results=classification_results,
        utility_results=utility_results,
        coefficient_table=coefficient_table,
    )

    print("\nExpanded-feature logistic-regression run completed successfully.")


if __name__ == "__main__":
    main()