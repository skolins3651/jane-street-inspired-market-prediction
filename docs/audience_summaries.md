# Audience Summaries

This document provides short summaries of the project for different reader types.

The goal is to make the project easier to discuss with non-technical readers, technical reviewers, recruiters, networking contacts, and quant-oriented audiences.

## General Short Summary

This project adapts the Jane Street Market Prediction Kaggle competition into a public-data quant research project. Using daily OHLCV data for a curated universe of liquid U.S. stocks, it predicts next-day stock performance relative to SPY and evaluates execute/pass decisions with an adapted utility score.

The project begins with simple baselines and logistic regression, then tests controlled improvements including threshold tuning, expanded features, nonlinear classification, and utility-aware action rules. The strongest result comes from reframing the task as a daily cross-sectional ranking problem: instead of classifying each stock-date row in isolation, the final model ranks stocks within each trading date and selects the top 8 of 30 opportunities.

## Non-Technical Summary

This project is about building a structured prediction system for financial markets.

The model looks at a group of large, liquid U.S. stocks each day and tries to identify which ones are most likely to do better than the overall market, represented by SPY, on the next trading day.

The most important part of the project is not just the final score. It is the process: defining the problem, building simple baselines, creating a scoring system, testing improvements one at a time, and explaining why the final model worked better than earlier attempts.

The key breakthrough was changing the question. Instead of asking only whether each stock would beat the market, the final model asked which stocks looked best compared with the other stocks available that same day.

## Recruiter / HR Summary

This is an end-to-end quantitative research project built in Python.

It includes data collection, feature engineering, chronological train/validation/test splits, baseline modeling, custom evaluation metrics, controlled model experiments, documentation, and final result interpretation.

The project is based on a public-data adaptation of the Jane Street Market Prediction Kaggle competition. The strongest model uses cross-sectional rank regression to select the top 8 stocks per day from a 30-stock universe, improving validation utility while remaining meaningfully selective.

This project demonstrates practical machine-learning workflow, financial modeling intuition, reproducible research habits, and the ability to explain technical results clearly.

## Technical Summary

This project predicts next-day stock performance relative to SPY using daily adjusted OHLCV data.

The baseline task is binary classification: for each stock-date row, predict whether the stock's next close-to-close return will exceed SPY's next close-to-close return. The project evaluates execute/pass decisions using an adapted Jane Street-style utility score based on selected weighted responses and daily consistency.

Initial models showed that simple binary classification produced weak or broad signals. Threshold tuning and utility-aware top-k selection improved validation utility, but largely by taking a large fraction of the universe.

The final model reframes the problem as cross-sectional rank regression. It trains on the daily percentile rank of future SPY-relative response and uses a constrained top-k selection rule. The selected rule takes the top 8 stocks per day, producing the strongest validation liquidity-weighted utility while keeping the action rate at 26.7%.

## Quant-Oriented Summary

This project is a public-data adaptation of a Jane Street-style market prediction problem.

The modeling universe consists of 30 liquid U.S. stocks, with SPY used as the market benchmark. The primary response is next-day stock return minus next-day SPY return. The initial target is a binary outperformance indicator, but the final model reframes the problem as daily cross-sectional rank prediction.

Evaluation uses an adapted utility score based on selected weighted responses aggregated by date, with liquidity-weighted utility as the main comparison metric and equal-weight utility as a sanity check. The project uses chronological train/validation/test splits and avoids test-set tuning.

The main modeling result is that cross-sectional rank regression outperformed earlier binary-classification and action-rule variants. The selected final rule takes the top 8 of 30 stocks per day, improving validation liquidity-weighted utility while maintaining a substantially lower action rate than the tuned-threshold logistic model.

The result should be interpreted as evidence of a promising research pipeline and ranking formulation, not as proof of a deployable trading strategy. The current version does not include transaction costs, slippage, market impact, walk-forward retraining, portfolio construction, or real P&L simulation.

## Networking Summary

I built a public-data quant research project inspired by the Jane Street Market Prediction Kaggle competition.

The project predicts next-day stock performance relative to SPY, evaluates trading-style execute/pass decisions, and compares several modeling approaches. The strongest result came from reframing the task as a cross-sectional ranking problem: instead of classifying stocks one by one, the model ranks the daily universe and selects the top opportunities.

I am using the project to demonstrate my ability to build a full research pipeline, reason through model design choices, and explain both results and limitations clearly.

## Application / Portfolio Summary

This project demonstrates an end-to-end quantitative modeling workflow:

* public market data acquisition;
* feature engineering;
* chronological train/validation/test design;
* baseline modeling;
* custom utility scoring;
* controlled improvement experiments;
* cross-sectional rank-regression modeling;
* result interpretation and limitations analysis.

The final model is not presented as a live trading strategy. It is presented as a research result: reframing the problem from binary classification to cross-sectional ranking produced the strongest and most selective validation performance.

## One-Sentence Version

A public-data, Jane Street-inspired market prediction project that shows how reframing a binary stock-outperformance classifier into a daily cross-sectional ranking model produced a stronger and more selective utility result.

## Two-Sentence Version

This project adapts the Jane Street Market Prediction Kaggle competition into a public-data quant research pipeline using daily OHLCV data for liquid U.S. stocks. After testing baselines, threshold tuning, expanded features, nonlinear classification, and utility-aware action rules, the strongest result came from a cross-sectional rank-regression model that selected the top 8 stocks per day.

## Longer Conversational Version

I built a quant research project inspired by the Jane Street Market Prediction Kaggle competition, but adapted it to public data because the original competition dataset is not available here.

The project uses daily market data for a curated universe of liquid U.S. stocks and asks which stocks are likely to outperform SPY over the next trading day. I started with simple baselines and logistic regression, built an adapted utility score, and then tested a series of controlled improvements.

The most interesting result was that the best model did not come from simply using a more complicated classifier. It came from changing the framing of the problem. Instead of predicting each stock-date row in isolation, the final model learns a daily cross-sectional ranking and selects the top 8 stocks per day. That produced the strongest validation result while staying meaningfully selective.