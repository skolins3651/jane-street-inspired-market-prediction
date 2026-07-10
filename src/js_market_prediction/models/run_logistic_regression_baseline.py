from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    / "logistic_regression_baseline_results.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_baseline_predictions.parquet"
)

COEFFICIENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_baseline_coefficients.csv"
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

WEIGHT_COLUMNS = [
    "weight_equal",
    "weight_liquidity",
]

SPLIT_ORDER = [
    "train",
    "validation",
    "test",
]

ACTION_THRESHOLD = 0.5


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
    """Validate the split-labeled feature dataset before modeling."""
    required_columns = {
        "date",
        "symbol",
        "split",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "weight_liquidity",
        *FEATURE_COLUMNS,
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

    if data[sorted(required_columns)].isna().sum().sum() > 0:
        raise ValueError(
            "Split dataset contains missing values in required columns."
        )

    if "SPY" in set(data["symbol"]):
        raise ValueError("SPY should not appear in the split dataset.")

    duplicate_rows = data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate date-symbol rows."
        )


def build_model() -> Pipeline:
    """Create the simple logistic-regression baseline model."""
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    return model


def fit_model(
    data: pd.DataFrame,
) -> Pipeline:
    """Fit the model using only the training split."""
    train_data = data.loc[
        data["split"] == "train"
    ].copy()

    features_train = train_data[FEATURE_COLUMNS]
    target_train = train_data["target_1d"].astype("int64")

    model = build_model()
    model.fit(
        features_train,
        target_train,
    )

    return model


def add_model_predictions(
    data: pd.DataFrame,
    model: Pipeline,
) -> pd.DataFrame:
    """Add predicted probabilities and binary actions."""
    prediction_data = data.sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True).copy()

    probabilities = model.predict_proba(
        prediction_data[FEATURE_COLUMNS]
    )[:, 1]

    prediction_data["probability_logistic_regression"] = probabilities

    prediction_data["action_logistic_regression"] = (
        prediction_data["probability_logistic_regression"]
        >= ACTION_THRESHOLD
    ).astype("int64")

    return prediction_data


def evaluate_model_predictions(
    prediction_data: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the logistic-regression baseline by split and weight."""
    result_rows = []

    for split_name in SPLIT_ORDER:
        split_data = prediction_data.loc[
            prediction_data["split"] == split_name
        ].copy()

        for weight_column in WEIGHT_COLUMNS:
            metrics = evaluate_actions(
                data=split_data,
                action_column="action_logistic_regression",
                weight_column=weight_column,
            )

            result_row = {
                "split": split_name,
                "model": "logistic_regression",
                "weight_column": weight_column,
                "action_threshold": ACTION_THRESHOLD,
                **metrics,
            }

            result_rows.append(result_row)

    results = pd.DataFrame(result_rows)

    return results


def extract_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """Extract standardized logistic-regression coefficients."""
    logistic_regression = model.named_steps[
        "logistic_regression"
    ]

    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": logistic_regression.coef_[0],
        }
    )

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return coefficients


def save_results(
    results: pd.DataFrame,
    output_path: Path = RESULTS_PATH,
) -> None:
    """Save model results to CSV."""
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


def save_predictions(
    prediction_data: pd.DataFrame,
    output_path: Path = PREDICTIONS_PATH,
) -> None:
    """Save row-level model predictions to Parquet."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_data.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Prediction file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Prediction file is empty: {output_path}"
        )


def save_coefficients(
    coefficients: pd.DataFrame,
    output_path: Path = COEFFICIENTS_PATH,
) -> None:
    """Save logistic-regression coefficients to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficients.to_csv(
        output_path,
        index=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Coefficient file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Coefficient file is empty: {output_path}"
        )


def print_model_summary(
    data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    results: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    """Print a compact summary of the ML baseline."""
    print("Logistic-regression baseline trained successfully.")
    print(f"Input path: {SPLIT_DATA_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"First date: {data['date'].min().date()}")
    print(f"Last date: {data['date'].max().date()}")

    print()
    print("Feature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print()
    print("Prediction summary by split:")
    prediction_summary = (
        prediction_data
        .groupby("split")
        .agg(
            row_count=("date", "size"),
            target_rate=("target_1d", "mean"),
            average_probability=(
                "probability_logistic_regression",
                "mean",
            ),
            action_rate=(
                "action_logistic_regression",
                "mean",
            ),
        )
        .loc[SPLIT_ORDER]
    )

    print(prediction_summary)

    print()
    print("Evaluation results:")
    display_columns = [
        "split",
        "model",
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
            ["split", "weight_column"]
        )
        .to_string(index=False)
    )

    print()
    print("Standardized coefficient ranking:")
    print(coefficients.to_string(index=False))

    print()
    print("Outputs saved successfully.")
    print(f"Results path: {RESULTS_PATH}")
    print(f"Predictions path: {PREDICTIONS_PATH}")
    print(f"Coefficients path: {COEFFICIENTS_PATH}")


def main() -> None:
    """Train and evaluate the simplest machine-learning baseline."""
    split_data = load_split_data()
    validate_split_data(split_data)

    model = fit_model(split_data)

    prediction_data = add_model_predictions(
        data=split_data,
        model=model,
    )

    results = evaluate_model_predictions(
        prediction_data=prediction_data,
    )

    coefficients = extract_coefficients(model)

    print_model_summary(
        data=split_data,
        prediction_data=prediction_data,
        results=results,
        coefficients=coefficients,
    )

    save_results(results)
    save_predictions(prediction_data)
    save_coefficients(coefficients)


if __name__ == "__main__":
    main()