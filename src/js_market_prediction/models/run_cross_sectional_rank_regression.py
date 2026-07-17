from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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

VALIDATION_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_rank_regression_validation_candidates.csv"
)

SELECTED_RULE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_rank_regression_selected.csv"
)

SELECTED_RULE_SCORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_rank_regression_scores.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_rank_regression_predictions.parquet"
)

MODEL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_rank_regression_model_summary.csv"
)

FEATURE_COLUMNS = ORIGINAL_FEATURE_COLUMNS + EXPANDED_FEATURE_COLUMNS

SPLIT_ORDER = ["train", "validation", "test"]
WEIGHT_COLUMNS = ["weight_equal", "weight_liquidity"]

TUNING_SPLIT = "validation"
MAIN_SELECTION_WEIGHT = "weight_liquidity"

TOP_K_CANDIDATES = [1, 2, 3, 5, 8, 10, 15]
RANDOM_STATE = 42


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

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Expanded feature dataset has missing values in required columns.")

    if not np.isfinite(data[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Expanded feature dataset has non-finite feature values.")

    observed_splits = set(data["split"].unique())
    expected_splits = set(SPLIT_ORDER)

    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, found {sorted(observed_splits)}"
        )


def add_cross_sectional_rank_target(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["resp_1d_rank_target"] = data.groupby("date")["resp_1d"].rank(
        method="average",
        pct=True,
    )

    if data["resp_1d_rank_target"].isna().any():
        raise ValueError("Rank target contains missing values.")

    if not data["resp_1d_rank_target"].between(0.0, 1.0).all():
        raise ValueError("Rank target must be between 0 and 1.")

    return data


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


def train_model(data: pd.DataFrame) -> HistGradientBoostingRegressor:
    train_data = data[data["split"] == "train"].copy()

    if train_data.empty:
        raise ValueError("Training split is empty.")

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    model.fit(
        train_data[FEATURE_COLUMNS],
        train_data["resp_1d_rank_target"],
    )

    return model


def add_predictions(
    data: pd.DataFrame,
    model: HistGradientBoostingRegressor,
) -> pd.DataFrame:
    data = data.copy()

    data["predicted_rank_score"] = model.predict(data[FEATURE_COLUMNS])

    return data


def add_top_k_action(data: pd.DataFrame, top_k: int) -> pd.DataFrame:
    data = data.copy()
    data["_original_index"] = data.index

    ranked = data.sort_values(
        by=["date", "predicted_rank_score", "symbol"],
        ascending=[True, False, True],
    ).copy()

    ranked["predicted_rank_within_date"] = ranked.groupby("date").cumcount() + 1
    ranked["action_cross_sectional_rank_regression"] = (
        ranked["predicted_rank_within_date"] <= top_k
    ).astype("int64")

    ranked = ranked.sort_values("_original_index").drop(columns="_original_index")
    ranked = ranked.reset_index(drop=True)

    return ranked


def calculate_action_diagnostics(data: pd.DataFrame) -> dict[str, float]:
    actual = data["target_1d"].astype(int)
    action = data["action_cross_sectional_rank_regression"].astype(int)

    true_positive = ((actual == 1) & (action == 1)).sum()
    false_positive = ((actual == 0) & (action == 1)).sum()
    false_negative = ((actual == 1) & (action == 0)).sum()

    predicted_positive = action.sum()
    actual_positive = actual.sum()

    accuracy = (actual == action).mean()
    action_rate = action.mean()

    if predicted_positive == 0:
        precision = float("nan")
        mean_rank_target_taken = float("nan")
        mean_predicted_score_taken = float("nan")
    else:
        precision = true_positive / (true_positive + false_positive)
        mean_rank_target_taken = data.loc[
            action == 1,
            "resp_1d_rank_target",
        ].mean()
        mean_predicted_score_taken = data.loc[
            action == 1,
            "predicted_rank_score",
        ].mean()

    if actual_positive == 0:
        recall = float("nan")
    else:
        recall = true_positive / (true_positive + false_negative)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "action_rate": action_rate,
        "mean_rank_target_taken": mean_rank_target_taken,
        "mean_predicted_score_taken": mean_predicted_score_taken,
    }


def calculate_ranking_diagnostics(data: pd.DataFrame) -> dict[str, float]:
    daily_spearman = data.groupby("date").apply(
        lambda group: group["predicted_rank_score"].corr(
            group["resp_1d_rank_target"],
            method="spearman",
        ),
        include_groups=False,
    )

    daily_spearman = daily_spearman.dropna()

    if daily_spearman.empty:
        mean_daily_spearman = float("nan")
        median_daily_spearman = float("nan")
        positive_spearman_rate = float("nan")
    else:
        mean_daily_spearman = daily_spearman.mean()
        median_daily_spearman = daily_spearman.median()
        positive_spearman_rate = (daily_spearman > 0).mean()

    return {
        "mean_daily_spearman": mean_daily_spearman,
        "median_daily_spearman": median_daily_spearman,
        "positive_spearman_rate": positive_spearman_rate,
    }


def score_validation_candidates(data: pd.DataFrame) -> pd.DataFrame:
    validation_data = data[data["split"] == TUNING_SPLIT].copy()

    if validation_data.empty:
        raise ValueError(f"No rows found for tuning split: {TUNING_SPLIT}")

    rows = []

    ranking_diagnostics = calculate_ranking_diagnostics(validation_data)

    for top_k in TOP_K_CANDIDATES:
        scoring_data = add_top_k_action(validation_data, top_k=top_k)
        action_diagnostics = calculate_action_diagnostics(scoring_data)

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                scoring_data,
                action_column="action_cross_sectional_rank_regression",
                weight_column=weight_column,
            )

            rows.append(
                {
                    "model": "cross_sectional_rank_regression",
                    "rule": "top_k_per_day",
                    "top_k": top_k,
                    "split": TUNING_SPLIT,
                    "weight_column": weight_column,
                    **ranking_diagnostics,
                    **action_diagnostics,
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


def score_selected_rule(
    data: pd.DataFrame,
    selected_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_data = add_top_k_action(data, top_k=selected_top_k)

    rows = []

    for split in SPLIT_ORDER:
        split_data = prediction_data[prediction_data["split"] == split].copy()

        if split_data.empty:
            raise ValueError(f"No rows found for split: {split}")

        ranking_diagnostics = calculate_ranking_diagnostics(split_data)
        action_diagnostics = calculate_action_diagnostics(split_data)

        for weight_column in WEIGHT_COLUMNS:
            utility_metrics = calculate_adapted_utility(
                split_data,
                action_column="action_cross_sectional_rank_regression",
                weight_column=weight_column,
            )

            rows.append(
                {
                    "model": "cross_sectional_rank_regression",
                    "rule": "top_k_per_day",
                    "top_k": selected_top_k,
                    "split": split,
                    "weight_column": weight_column,
                    **ranking_diagnostics,
                    **action_diagnostics,
                    **utility_metrics,
                }
            )

    return prediction_data, pd.DataFrame(rows)


def build_model_summary(model: HistGradientBoostingRegressor) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "cross_sectional_rank_regression",
                "model_class": model.__class__.__name__,
                "feature_count": len(FEATURE_COLUMNS),
                "target": "daily cross-sectional percentile rank of resp_1d",
                "max_top_k_candidate": max(TOP_K_CANDIDATES),
                "max_action_rate_candidate": max(TOP_K_CANDIDATES) / 30,
                "max_iter": model.max_iter,
                "learning_rate": model.learning_rate,
                "max_leaf_nodes": model.max_leaf_nodes,
                "min_samples_leaf": model.min_samples_leaf,
                "l2_regularization": model.l2_regularization,
                "early_stopping": model.early_stopping,
                "random_state": RANDOM_STATE,
            }
        ]
    )


def save_outputs(
    validation_results: pd.DataFrame,
    selected_rule: pd.DataFrame,
    selected_rule_scores: pd.DataFrame,
    prediction_data: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    VALIDATION_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)

    validation_results.to_csv(VALIDATION_CANDIDATES_PATH, index=False)
    selected_rule.to_csv(SELECTED_RULE_PATH, index=False)
    selected_rule_scores.to_csv(SELECTED_RULE_SCORES_PATH, index=False)
    model_summary.to_csv(MODEL_SUMMARY_PATH, index=False)

    prediction_columns = [
        "date",
        "symbol",
        "split",
        "target_1d",
        "resp_1d",
        "resp_1d_rank_target",
        "weight_equal",
        "weight_liquidity",
        "predicted_rank_score",
        "predicted_rank_within_date",
        "action_cross_sectional_rank_regression",
    ]

    prediction_data[prediction_columns].to_parquet(PREDICTIONS_PATH, index=False)


def print_summary(
    validation_results: pd.DataFrame,
    selected_rule: pd.DataFrame,
    selected_rule_scores: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    summary_columns = [
        "model",
        "rule",
        "top_k",
        "split",
        "weight_column",
        "utility",
        "total_profit",
        "c_stat",
        "action_rate",
        "mean_resp_taken",
        "mean_rank_target_taken",
        "mean_daily_spearman",
        "positive_spearman_rate",
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

    print("\nModel summary:")
    print(model_summary.to_string(index=False))

    print(f"\nSaved validation candidates to: {VALIDATION_CANDIDATES_PATH}")
    print(f"Saved selected rule to:         {SELECTED_RULE_PATH}")
    print(f"Saved selected rule scores to:  {SELECTED_RULE_SCORES_PATH}")
    print(f"Saved predictions to:           {PREDICTIONS_PATH}")
    print(f"Saved model summary to:         {MODEL_SUMMARY_PATH}")


def main() -> None:
    data = load_expanded_data(EXPANDED_DATA_PATH)
    data = add_cross_sectional_rank_target(data)
    validate_daily_symbol_counts(data)

    model = train_model(data)
    prediction_data = add_predictions(data, model)

    validation_results = score_validation_candidates(prediction_data)
    selected_rule = select_best_rule(validation_results)

    selected_top_k = int(selected_rule["top_k"].iloc[0])
    selected_prediction_data, selected_rule_scores = score_selected_rule(
        prediction_data,
        selected_top_k=selected_top_k,
    )

    model_summary = build_model_summary(model)

    save_outputs(
        validation_results=validation_results,
        selected_rule=selected_rule,
        selected_rule_scores=selected_rule_scores,
        prediction_data=selected_prediction_data,
        model_summary=model_summary,
    )

    print_summary(
        validation_results=validation_results,
        selected_rule=selected_rule,
        selected_rule_scores=selected_rule_scores,
        model_summary=model_summary,
    )

    print("\nCross-sectional rank-regression run completed successfully.")


if __name__ == "__main__":
    main()