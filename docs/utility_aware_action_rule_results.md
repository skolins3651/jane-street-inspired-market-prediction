# Utility-Aware Action Rule Results

This document summarizes the utility-aware action-rule experiment.

## Purpose

The earlier threshold-tuning experiment showed that the default `0.50` classification threshold was not necessarily the best action rule for the adapted utility framework.

This experiment tested a different decision rule: rather than taking every stock above a fixed probability threshold, the model ranks stocks by predicted probability each day and takes only the top `k` opportunities.

The experiment keeps the original logistic-regression model, original feature set, prediction target, chronological split structure, and adapted utility scoring framework fixed. It changes only the action rule.

## Method

For each trading day, stocks were ranked by their logistic-regression predicted probability.

The tested rules were:

```text
Take the top k stocks per day, where k is one of:
1, 2, 3, 5, 8, 10, 15, 20, 25
```

The best value of `k` was selected using validation liquidity-weighted utility.

Equal-weight utility and companion diagnostics were also checked to evaluate whether the result was robust or pathological.

## Selected Rule

The best rule by validation liquidity-weighted utility was:

```text
Take the top 20 stocks per day.
```

Since the tradable universe contains 30 stocks per day, this rule takes approximately two-thirds of available opportunities.

| Rule | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate | Mean Response Taken |
|---|---:|---:|---:|---:|
| Top 20 per day | 0.132720 | 4.248294 | 0.666667 | 0.000057 |

## Comparison to Earlier Results

| Model / Rule | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate |
|---|---:|---:|---:|
| Original logistic, threshold 0.50 | 0.000000 | 2.870843 | 0.281836 |
| Expanded-feature logistic, threshold 0.50 | 0.175228 | 2.670977 | 0.291550 |
| Gradient boosting, threshold 0.50 | 0.000000 | 0.468004 | 0.167265 |
| Utility-aware top-20/day rule | 0.132720 | 4.248294 | 0.666667 |
| Tuned-threshold logistic, threshold 0.48 | 1.289191 | 6.283775 | 0.876114 |

## Validation Candidate Pattern

The smaller top-<var>k</var> rules generally performed poorly.

The best utility-aware rule was not highly selective. It selected 20 out of 30 stocks per day, while rules selecting only the top 1, 2, 3, 5, 8, or 10 stocks were weak or negative under equal-weight utility.

This suggests that the logistic-regression model has some useful broad ranking information, but it is not especially good at identifying only the very best few opportunities each day.

## Train, Validation, and Test Behavior

| Split | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate | Mean Response Taken |
|---|---:|---:|---:|---:|
| Train | 13.548172 | 14.864507 | 0.666667 | 0.000311 |
| Validation | 0.132720 | 4.248294 | 0.666667 | 0.000057 |
| Test | 0.000000 | 2.992545 | 0.666667 | -0.000141 |

The selected rule has positive liquidity-weighted utility on validation and test, but its equal-weight behavior is weak. On the test split, equal-weight total profit is negative even though liquidity-weighted utility remains positive.

This reinforces the importance of checking both weighting schemes.

## Interpretation

The utility-aware top-<var>k</var> rule improved validation liquidity-weighted utility relative to the original logistic-regression baseline, the expanded-feature logistic model, and the gradient-boosting baseline.

However, it did not outperform the tuned-threshold logistic rule. The tuned threshold remains the strongest validation result so far.

The result is still informative. It suggests that the logistic-regression probabilities contain some useful ranking signal, especially when the rule takes a broad portion of the daily universe.

At the same time, the result should be interpreted cautiously. This experiment directly optimizes an action rule around the adapted utility score, so it is more score-aware than the earlier experiments. Because the utility score is a project-defined approximation rather than a real trading P&L system, this rule should not be treated as proof of a profitable strategy.

Instead, it should be treated as evidence about the modeling pipeline:

* action-rule design matters;
* the model's probability ranking contains some information;
* the ranking is not strong enough to confidently isolate only the top few names;
* liquidity-weighted and equal-weight conclusions can diverge;
* utility improvements need companion diagnostics.

## Main Lesson

The utility-aware action rule produced a meaningful improvement over the default logistic-regression action rule, but it did not become the best overall result.

The strongest current candidate remains the tuned-threshold logistic rule. The top-<var>k</var> experiment is valuable because it shows that better probability ranking could be a promising direction for future work.

A stronger model that ranks opportunities more effectively might improve utility substantially, especially if it can identify a smaller set of high-quality daily opportunities without relying on a broad action rate.
