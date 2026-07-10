# Baseline Metrics

This document defines the initial metrics used to evaluate baseline strategies and simple models.

The goal is not yet to reproduce the full Jane Street utility score. Instead, these metrics provide simple classification and trading sanity checks for the one-day SPY-relative outperformance problem.

## Prediction Setup

Each model or baseline strategy produces an action for each stock-date row:

```text
action = 1 means take the long stock opportunity
action = 0 means pass
```

The realized binary target is:

```text
target_1d = 1 if resp_1d > 0 else 0
target_1d = 0 otherwise
```

A positive `resp_1d` means the stock outperformed SPY over the next close-to-close trading interval.

## Dataset Diagnostics

Each split report should include:

```text
row_count
trading_date_count
symbol_count
target_rate
```

These diagnostics confirm the size and class balance of the train, validation, and test splits.

## Classification Metrics

The initial classification metrics are:

```text
accuracy
precision
recall
predicted_positive_rate
```

### Accuracy

Accuracy measures the fraction of rows where the predicted action matches the binary target.

```text
accuracy = mean(action == target_1d)
```

This is easy to understand, but it is not sufficient by itself because a trading strategy may care more about the quality of taken opportunities than about correctly passing on unattractive ones.

### Precision

Precision measures how often taken opportunities were correct.

```text
precision = true_positives / predicted_positives
```

In this project, precision can be interpreted as the hit rate among rows where the strategy chose `action = 1`.

### Recall

Recall measures how many outperforming opportunities were captured.

```text
recall = true_positives / actual_positives
```

Recall is useful for understanding whether a strategy captures many positive opportunities or only a small, selective subset.

### Predicted Positive Rate

Predicted positive rate measures how often the strategy takes an action.

```text
predicted_positive_rate = mean(action)
```

This is important because an always-take strategy and a highly selective strategy can have very different economic meanings even if their classification metrics look similar.

## Trading Sanity-Check Metrics

The initial trading sanity-check metrics are:

```text
mean_resp_taken
weighted_mean_resp_taken
total_weighted_resp_taken
weighted_hit_rate_taken
```

These metrics are only calculated on rows where `action = 1`.

### Mean Response Taken

Mean response taken measures the average realized excess return for taken opportunities.

```text
mean_resp_taken = mean(resp_1d where action = 1)
```

A positive value means that, on average, the selected stocks outperformed SPY.

### Weighted Mean Response Taken

Weighted mean response taken uses an observation-weight column, such as `weight_equal` or `weight_liquidity`.

```text
weighted_mean_resp_taken =
    sum(weight * resp_1d * action) / sum(weight * action)
```

This allows equal-weight and liquidity-weighted evaluations to be compared.

### Total Weighted Response Taken

Total weighted response taken is a simple aggregate payoff-style metric.

```text
total_weighted_resp_taken =
    sum(weight * resp_1d * action)
```

This is not yet the final Jane Street-style utility score, but it gives a useful directional check of whether a strategy accumulates positive or negative weighted excess return.

### Weighted Hit Rate Taken

Weighted hit rate taken measures the weighted fraction of taken opportunities where `target_1d = 1`.

```text
weighted_hit_rate_taken =
    sum(weight * target_1d * action) / sum(weight * action)
```

This is a weighted version of precision.

## Weight Columns

Trading sanity-check metrics are reported using both available weight columns when useful:

```text
weight_equal
weight_liquidity
```

`weight_equal` is the clearest default for first-pass interpretation. `weight_liquidity` checks whether conclusions change when more liquid observations receive more evaluation weight.

## Baseline Strategies to Compare Later

The metrics in this document will be used for:

```text
always-pass
always-take
simple rule-based strategy
simple machine-learning baseline
```

The always-pass strategy is useful as a classification reference point, but trading metrics that require taken opportunities may be undefined for always-pass because it never chooses `action = 1`.

More specifically, for any strategies that never take an opportunity, denominator-based metrics requiring taken rows or predicted positives are reported as undefined rather than forced to zero. This applies to precision, mean response taken, weighted mean response taken, and weighted hit rate taken when their denominators are zero. `total_weighted_resp_taken` is still defined for always-pass and equals `0`, because no opportunities are taken.

## Deferred Metrics

The following are intentionally deferred:

* Jane Street-style utility score
* transaction-cost-adjusted returns
* slippage-adjusted returns
* portfolio-level capital constraints
* probability calibration metrics
* threshold optimization metrics

These may be added in future extensions after the simple baseline framework is working.