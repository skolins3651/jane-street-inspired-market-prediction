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

VALIDATION_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "utility_aware_action_rule_validation_candidates.csv"
)

SELECTED_RULE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "utility_aware_action_rule_selected.csv"
)

SELECTED_RULE_SCORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "utility_aware_action_rule_scores.csv"
)

SELECTED_RULE_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "utility_aware_action_rule_predictions.parquet"
)

SPLIT_ORDER = ["train", "validation", "test"]
WEIGHT_COLUMNS = ["weight_equal", "weight_liquidity"]

TUNING_SPLIT = "validation"
MAIN_SELECTION_WEIGHT = "weight_liquidity"

TOP_K_CANDIDATES = [1, 2, 3, 5, 8, 10, 15, 20, 25]


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split dataset not found: {path}")

    data = pd.read_parquet(path)
    validate_split_data(data)
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_logistic_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Logistic-regression predictions not found: {path}\n"
            "Run this first:\n"
            "python -m js_market_prediction.models.run_logistic_regression_baseline"
        )

    predictions = pd.read_parquet(path)
    validate_logistic_predictions(predictions)
    return predictions.sort_values(["date", "symbol"]).reset_index(drop=True)


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

    if data[["date", "symbol"]].duplicated().any():
        raise ValueError("Split dataset has duplicate date-symbol rows.")

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


def validate_daily_symbol_counts(data: pd.DataFrame) -> None:
    counts = data.groupby(["split", "date"])["symbol"].nunique()

    min_count = counts.min()
    max_count = counts.max()

    if min_count != max_count:
        raise ValueError(
            "Daily symbol counts are not constant. "
            f"Minimum count: {min_count}, maximum count: {max_count}"
        )

    max_candidate = max(TOP_K_CANDIDATES)

    if max_candidate > min_count:
        raise ValueError(
            f"Top-k candidate {max_candidate} exceeds daily symbol count {min_count}."
        )


def add_top_k_action(data: pd.DataFrame, top_k: int) -> pd.DataFrame:
    data = data.copy()
    data["_original_index"] = data.index

    ranked = data.sort_values(
        by=["date", "probability_logistic_regression", "symbol"],
        ascending=[True, False, True],
    ).copy()

    ranked["probability_rank_within_date"] = ranked.groupby("date").cumcount() + 1
    ranked["action_utility_aware"] = (
        ranked["probability_rank_within_date"] <= top_k
    ).astype("int64")

    ranked = ranked.sort_values("_original_index").drop(columns="_original_index")
    ranked = ranked.reset_index(drop=True)

    return ranked


def calculate_classification_metrics(data: pd.DataFrame) -> dict[str, float]:
    actual = data["target_1d"].astype(int)
    action = data["action_utility_aware"].astype(int)

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


def score_validation_candidates(data: pd.DataFrame) -> pd.DataFrame:
    validation_data = data[data["split"] == TUNING_SPLIT].copy()

    if validation_data.empty:
        raise ValueError(f"No rows found for tuning split: {TUNING_SPLIT}")

    rows = []

    for top_k in TOP_K_CANDIDATES:
        scoring_data = add_top_k_action(validation_data, top_k=top_k)
        classification_metrics = calculate_classification_metrics(scoring_data)

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                scoring_data,
                action_column="action_utility_aware",
                weight_column=weight_column,
            )

            rows.append(
                {
                    "rule": "top_k_per_day",
                    "top_k": top_k,
                    "split": TUNING_SPLIT,
                    "weight_column": weight_column,
                    **classification_metrics,
                    **utility_metrics,
                }
            )

    return pd.DataFrame(rows)


def select_best_rule(validation_results: pd.DataFrame) -> pd.DataFrame:
    selection_data = validation_results[
        validation_results["weight_column"] == MAIN_SELECTION_WEIGHT
    ].copy()

    if selection_data.empty:
        raise ValueError(
            f"No validation results found for weight column: {MAIN_SELECTION_WEIGHT}"
        )

    selection_data = selection_data.sort_values(
        by=["utility", "total_profit", "mean_resp_taken", "action_rate"],
        ascending=[False, False, False, True],
    )

    return selection_data.head(1)


def score_selected_rule(data: pd.DataFrame, selected_top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_data = add_top_k_action(data, top_k=selected_top_k)

    rows = []

    for split in SPLIT_ORDER:
        split_data = prediction_data[prediction_data["split"] == split].copy()

        if split_data.empty:
            raise ValueError(f"No rows found for split: {split}")

        classification_metrics = calculate_classification_metrics(split_data)

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                split_data,
                action_column="action_utility_aware",
                weight_column=weight_column,
            )

            rows.append(
                {
                    "rule": "top_k_per_day",
                    "top_k": selected_top_k,
                    "split": split,
                    "weight_column": weight_column,
                    **classification_metrics,
                    **utility_metrics,
                }
            )

    return prediction_data, pd.DataFrame(rows)


def save_outputs(
    validation_results: pd.DataFrame,
    selected_rule: pd.DataFrame,
    selected_rule_scores: pd.DataFrame,
    prediction_data: pd.DataFrame,
) -> None:
    VALIDATION_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)

    validation_results.to_csv(VALIDATION_CANDIDATES_PATH, index=False)
    selected_rule.to_csv(SELECTED_RULE_PATH, index=False)
    selected_rule_scores.to_csv(SELECTED_RULE_SCORES_PATH, index=False)

    prediction_columns = [
        "date",
        "symbol",
        "split",
        "target_1d",
        "resp_1d",
        "weight_equal",
        "weight_liquidity",
        "probability_logistic_regression",
        "probability_rank_within_date",
        "action_utility_aware",
    ]

    prediction_data[prediction_columns].to_parquet(
        SELECTED_RULE_PREDICTIONS_PATH,
        index=False,
    )


def print_summary(
    validation_results: pd.DataFrame,
    selected_rule: pd.DataFrame,
    selected_rule_scores: pd.DataFrame,
) -> None:
    summary_columns = [
        "rule",
        "top_k",
        "split",
        "weight_column",
        "utility",
        "total_profit",
        "c_stat",
        "action_rate",
        "mean_resp_taken",
    ]

    print("\nValidation candidate results:")
    print(
        validation_results[summary_columns]
        .sort_values(["weight_column", "top_k"])
        .to_string(index=False)
    )

    print("\nSelected rule using validation liquidity-weighted utility:")
    print(selected_rule[summary_columns].to_string(index=False))

    print("\nSelected rule scores across splits:")
    print(
        selected_rule_scores[summary_columns]
        .sort_values(["split", "weight_column"])
        .to_string(index=False)
    )

    print(f"\nSaved validation candidates to: {VALIDATION_CANDIDATES_PATH}")
    print(f"Saved selected rule to:         {SELECTED_RULE_PATH}")
    print(f"Saved selected rule scores to:  {SELECTED_RULE_SCORES_PATH}")
    print(f"Saved selected predictions to:  {SELECTED_RULE_PREDICTIONS_PATH}")


def main() -> None:
    split_data = load_split_data(SPLIT_DATA_PATH)
    logistic_predictions = load_logistic_predictions(LOGISTIC_PREDICTIONS_PATH)

    data = merge_predictions(split_data, logistic_predictions)
    validate_daily_symbol_counts(data)

    validation_results = score_validation_candidates(data)
    selected_rule = select_best_rule(validation_results)

    selected_top_k = int(selected_rule["top_k"].iloc[0])
    prediction_data, selected_rule_scores = score_selected_rule(
        data,
        selected_top_k=selected_top_k,
    )

    save_outputs(
        validation_results=validation_results,
        selected_rule=selected_rule,
        selected_rule_scores=selected_rule_scores,
        prediction_data=prediction_data,
    )

    print_summary(
        validation_results=validation_results,
        selected_rule=selected_rule,
        selected_rule_scores=selected_rule_scores,
    )

    print("\nUtility-aware action-rule run completed successfully.")


if __name__ == "__main__":
    main()