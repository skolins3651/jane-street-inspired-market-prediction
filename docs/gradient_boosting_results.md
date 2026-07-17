# Gradient Boosting Results

This document summarizes the gradient-boosting baseline experiment.

## Purpose

The original machine-learning baseline used logistic regression with the initial 9-feature dataset.

Logistic regression is useful because it is simple and interpretable, but it can only model linear relationships after feature scaling. This experiment tested whether a simple tree-based model could improve results by capturing nonlinear relationships and feature interactions.

The experiment changed only the model family. It kept the prediction target, original feature set, chronological split structure, default `0.50` action threshold, and adapted utility scoring framework fixed.

## Method

The model used `HistGradientBoostingClassifier` from scikit-learn with the original 9-feature dataset.

The action rule was:

```text
action = 1 if predicted_probability >= 0.50 else 0
```

The model was trained on the training split and evaluated on train, validation, and test splits. The main comparison score remains validation liquidity-weighted utility.

## Validation Results

| Model / Rule | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate | Mean Response Taken |
|---|---:|---:|---:|---:|
| Original logistic, threshold 0.50 | 0.000000 | 2.870843 | 0.281836 | -0.000298 |
| Expanded-feature logistic, threshold 0.50 | 0.175228 | 2.670977 | 0.291550 | 0.000127 |
| Tuned-threshold logistic, threshold 0.48 | 1.289191 | 6.283775 | 0.876114 | 0.000145 |
| Gradient boosting, threshold 0.50 | 0.000000 | 0.468004 | 0.167265 | -0.000208 |

## Gradient-Boosting Model Details

On the validation split, the gradient-boosting model had:

| Metric | Value |
|---|---:|
| Accuracy | 0.502528 |
| Precision | 0.499602 |
| Recall | 0.168027 |
| Action rate | 0.167265 |
| Equal-weight utility | 0.000000 |
| Liquidity-weight utility | 0.468004 |
| Equal-weight total profit | -0.523475 |
| Liquidity-weight total profit | 1.277750 |
| Mean response taken | -0.000208 |

## Train vs. Validation Behavior

The model performed much better on the training split than on the validation split.

| Split | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate | Mean Response Taken |
|---|---:|---:|---:|---:|
| Train | 52.972661 | 49.098949 | 0.120143 | 0.002591 |
| Validation | 0.000000 | 0.468004 | 0.167265 | -0.000208 |

This gap suggests that the tree-based model learned relationships in the training period that did not transfer well to validation.

The result does not necessarily mean that tree-based models are unsuitable for this problem. It means that this constrained tree-based baseline, using the original feature set and default threshold, did not improve validation performance.

## Interpretation

The gradient-boosting model was more selective than the original logistic-regression baseline, with a validation action rate of about `16.7%`.

However, greater selectivity did not translate into better selected opportunities. The model's validation mean response taken was negative, and its equal-weight total profit was negative.

The liquidity-weighted utility was positive, but substantially lower than the original logistic-regression baseline and far below the tuned-threshold logistic result.

This suggests that nonlinear modeling alone did not solve the main weakness of the baseline framework.

## Main Lesson

Changing the model family did not improve the main validation score.

The experiment is still useful because it rules out a simple explanation: the initial model was not weak merely because logistic regression was too linear. Under the current target, feature set, and default threshold, the tree-based model showed signs of overfitting and did not generalize strongly to validation.

The next controlled experiment should focus on the action rule rather than the model family. Classification defaults do not necessarily match the adapted utility objective, and the threshold-tuning experiment already showed that action-rule changes can materially affect utility.