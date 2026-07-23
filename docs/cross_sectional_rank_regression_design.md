# Cross-Sectional Rank-Regression Design

This document defines the final scoped modeling extension for the Jane Street–inspired market prediction project.

## Motivation

The planned improvement experiments produced mixed results.

The strongest improvements came from changing the action rule rather than from changing the feature set or model family alone. Threshold tuning and the utility-aware top-<var>k</var> rule both improved validation liquidity-weighted utility, suggesting that the original logistic-regression probabilities contain some useful broad signal.

However, the best-performing rules were not highly selective. The tuned-threshold logistic model took most available opportunities, and the utility-aware top-<var>k</var> rule performed best when taking 20 out of 30 stocks per day.

This suggests a specific weakness: the existing models are better at identifying broad exposure than at ranking the best few opportunities.

## Ranking and Selection

The project can separate the decision problem into two stages.

The ranking stage tries to estimate which stocks are more attractive relative to the other stocks available on the same date.

The selection stage decides which ranked opportunities to take.

The earlier action-rule experiments mostly improved the selection stage. The missing question is whether a model can improve the ranking stage itself.

## Model Idea

The cross-sectional rank-regression model reframes the problem.

Instead of predicting only whether a stock will outperform SPY, the model attempts to predict each stock's future relative rank within the daily stock universe.

The training target is based on `resp_1d`, ranked cross-sectionally by date. Stocks with higher next-day SPY-relative returns receive higher rank targets for that date.

This target better matches the desired use case: selecting relatively attractive opportunities from the daily universe.

## Feature Set and Model Choice

The model uses the expanded feature set.

This is a scoped extension rather than a controlled single-change experiment. The rationale is that the earlier experiments showed:

* action-rule design mattered most;
* expanded features improved some broad diagnostics but did not improve the main score by themselves;
* nonlinear classification did not generalize well using the original feature set;
* the remaining weakness appears to be selective ranking.

A tree-based regressor is appropriate because it can model nonlinear relationships and interactions among the expanded features while implicitly down-weighting unhelpful splits.

The project will not perform a broad search over model families, feature subsets, or hyperparameters.

## Selection Rule

Predicted rank scores will be converted into actions using a constrained daily top-<var>k</var> rule.

The candidate values are:

```text
1, 2, 3, 5, 8, 10, 15
```

Because the daily tradable universe contains 30 stocks, the maximum candidate action rate is 50%.

This constraint is intentional. A successful result should not merely take most of the universe. It should show evidence of selective discrimination.

## Success Criteria

The model will be considered promising if it shows:

* positive validation liquidity-weighted utility;
* positive validation equal-weight utility;
* positive validation mean response taken;
* action rate no higher than 50%;
* reasonable train-validation behavior;
* improvement over the original logistic-regression baseline on selective diagnostics.

The model does not need to beat every prior rule on liquidity-weighted utility to be informative. A more selective model with positive equal-weight utility and positive mean response taken may provide a stronger analytical signal than a higher-utility rule that takes most available opportunities.

## Interpretation Risk

This experiment is still not proof of a profitable trading strategy.

The data is based on public daily OHLCV features, the universe is small, and the utility score is an adapted research metric rather than real trading P&L.

The goal is to test whether the pipeline can learn a more meaningful cross-sectional ranking signal under these constraints.
