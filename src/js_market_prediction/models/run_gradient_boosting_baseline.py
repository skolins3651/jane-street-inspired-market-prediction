from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from js_market_prediction.evaluation.utility import calculate_adapted_utility


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_and_splits_1d.parquet"
)

CLASSIFICATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gradient_boosting_baseline_results.csv"
)

UTILITY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gradient_boosting_baseline_utility_scores.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gradient_boosting_baseline_predictions.parquet"
)

MODEL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gradient_boosting_baseline_model_summary.csv"
)

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "intraday_range",
    "overnight_gap",
    "log_volume",
    "log_dollar_volume",
    "relative_volume_20d",
]

SPLIT_ORDER = ["train", "validation", "test"]
WEIGHT_COLUMNS = ["weight_equal", "weight_liquidity"]

ACTION_THRESHOLD = 0.50
RANDOM_STATE = 42


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Split dataset not found: {path}\n"
            "Run this first:\n"
            "python -m js_market_prediction.data.build_splits"
        )

    data = pd.read_parquet(path)
    validate_split_data(data)

    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def validate_split_data(data: pd.DataFrame) -> None:
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
        raise ValueError(f"Split dataset is missing columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Split dataset is empty.")

    if data[["date", "symbol"]].duplicated().any():
        raise ValueError("Split dataset has duplicate date-symbol rows.")

    required_data = data[list(required_columns)]

    if required_data.isna().any().any():
        raise ValueError("Split dataset has missing values in required columns.")

    if not np.isfinite(data[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Split dataset has non-finite feature values.")

    observed_splits = set(data["split"].unique())
    expected_splits = set(SPLIT_ORDER)

    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, found {sorted(observed_splits)}"
        )


def train_model(data: pd.DataFrame) -> HistGradientBoostingClassifier:
    train_data = data[data["split"] == "train"].copy()

    if train_data.empty:
        raise ValueError("Training split is empty.")

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )

    model.fit(train_data[FEATURE_COLUMNS], train_data["target_1d"].astype(int))

    return model


def add_predictions(
    data: pd.DataFrame,
    model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    data = data.copy()

    probabilities = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]

    data["probability_gradient_boosting"] = probabilities
    data["action_gradient_boosting"] = (
        data["probability_gradient_boosting"] >= ACTION_THRESHOLD
    ).astype("int64")

    return data


def calculate_classification_metrics(data: pd.DataFrame) -> dict[str, float]:
    actual = data["target_1d"].astype(int)
    action = data["action_gradient_boosting"].astype(int)

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
                "model": "gradient_boosting_baseline",
                "split": split,
                "threshold": ACTION_THRESHOLD,
                **classification_metrics,
            }
        )

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                split_data,
                action_column="action_gradient_boosting",
                weight_column=weight_column,
            )

            utility_rows.append(
                {
                    "model": "gradient_boosting_baseline",
                    "split": split,
                    "threshold": ACTION_THRESHOLD,
                    "weight_column": weight_column,
                    **classification_metrics,
                    **utility_metrics,
                }
            )

    return pd.DataFrame(classification_rows), pd.DataFrame(utility_rows)


def build_model_summary(model: HistGradientBoostingClassifier) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "gradient_boosting_baseline",
                "model_class": model.__class__.__name__,
                "feature_count": len(FEATURE_COLUMNS),
                "threshold": ACTION_THRESHOLD,
                "max_iter": model.max_iter,
                "learning_rate": model.learning_rate,
                "max_leaf_nodes": model.max_leaf_nodes,
                "min_samples_leaf": model.min_samples_leaf,
                "l2_regularization": model.l2_regularization,
                "random_state": RANDOM_STATE,
            }
        ]
    )


def save_outputs(
    prediction_data: pd.DataFrame,
    classification_results: pd.DataFrame,
    utility_results: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    CLASSIFICATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    classification_results.to_csv(CLASSIFICATION_RESULTS_PATH, index=False)
    utility_results.to_csv(UTILITY_RESULTS_PATH, index=False)
    model_summary.to_csv(MODEL_SUMMARY_PATH, index=False)

    prediction_columns = [
        "date",
        "symbol",
        "split",
        "target_1d",
        "resp_1d",
        "weight_equal",
        "weight_liquidity",
        "probability_gradient_boosting",
        "action_gradient_boosting",
    ]

    prediction_data[prediction_columns].to_parquet(PREDICTIONS_PATH, index=False)


def print_summary(
    prediction_data: pd.DataFrame,
    classification_results: pd.DataFrame,
    utility_results: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    print("\nGradient-boosting prediction summary:")
    prediction_summary = prediction_data.groupby("split").agg(
        row_count=("target_1d", "size"),
        target_rate=("target_1d", "mean"),
        average_probability=("probability_gradient_boosting", "mean"),
        action_rate=("action_gradient_boosting", "mean"),
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

    print("\nModel summary:")
    print(model_summary.to_string(index=False))

    print(f"\nSaved classification results to: {CLASSIFICATION_RESULTS_PATH}")
    print(f"Saved utility results to:        {UTILITY_RESULTS_PATH}")
    print(f"Saved predictions to:            {PREDICTIONS_PATH}")
    print(f"Saved model summary to:          {MODEL_SUMMARY_PATH}")


def main() -> None:
    data = load_split_data(SPLIT_DATA_PATH)

    model = train_model(data)
    prediction_data = add_predictions(data, model)

    classification_results, utility_results = evaluate_model(prediction_data)
    model_summary = build_model_summary(model)

    save_outputs(
        prediction_data=prediction_data,
        classification_results=classification_results,
        utility_results=utility_results,
        model_summary=model_summary,
    )

    print_summary(
        prediction_data=prediction_data,
        classification_results=classification_results,
        utility_results=utility_results,
        model_summary=model_summary,
    )

    print("\nGradient-boosting baseline run completed successfully.")


if __name__ == "__main__":
    main()