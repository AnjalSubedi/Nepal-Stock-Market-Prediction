# 🇳🇵 Nepal Stock Market Analysis & Prediction (2012–2020)

A comprehensive end-to-end machine learning project on NEPSE (Nepal Stock Exchange) data — covering exploratory data analysis, feature engineering, and three prediction tasks.

## 📊 Dataset
- **Source:** [Nepali Stock Market Dataset (Kaggle)](https://www.kaggle.com/datasets/sagyamthapa/nepali-stock-market-form-2012-to-2020-till-march)
- **Period:** 2012 – March 2020
- **Records:** 249,000+ daily OHLCV trading records across multiple stocks
- **Focus Stock:** Standard Chartered Bank Nepal (SCB)

## 🎯 Prediction Tasks

| Task | Type | Best Model | Score |
|------|------|-----------|-------|
| **Task A** — Tomorrow's Close Price | Regression | Linear Regression | RMSE: 22.45, R² = 0.851 |
| **Task B** — Tomorrow's Return (%) | Regression | Random Forest | R² = 0.095 |
| **Task C** — UP or DOWN Direction | Classification | Random Forest (tuned) | ROC-AUC = 0.618 |

## 🔧 Features Engineered (139 total)
- Return & lag features (1, 2, 3, 5, 7, 14, 30 day lags)
- Rolling statistics (mean, std, skew) across multiple windows
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV
- Volume ratios and anomaly features
- Market-wide context (breadth, advancers ratio, relative return)
- Cyclical time encoding (month, day-of-week)

## 🤖 Models Used
- Linear Regression & Ridge Regression (CV-tuned via `TimeSeriesSplit`)
- Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning via `RandomizedSearchCV` with time-series-safe cross-validation

## 📈 Key Insights
- Raw price prediction: baseline (yesterday's close) is hard to beat — consistent with the **Efficient Market Hypothesis**
- Return prediction: all ML models beat the naive baseline; tree models work well on stationary features
- Direction prediction: Random Forest achieves AUC = 0.618 — **significantly above random (0.5)**
- Top predictors: market breadth, volume anomalies, relative return, short-term momentum

## 🗂️ Project Structure
```
nepal-stock-market.ipynb   ← Main notebook (EDA + 3 ML tasks)
nepal-stock-lstm-mtl.ipynb ← LSTM Multi-Task Learning (run on Kaggle GPU)
Data/
  OHLC.csv                 ← Raw OHLCV data
```

## 🚀 LSTM Extension
A separate notebook (`nepal-stock-lstm-mtl.ipynb`) implements an **LSTM Multi-Task Learning** model trained on **all NEPSE stocks simultaneously** with:
- Stock embeddings (each stock gets a learned representation)
- Shared LSTM layers + task-specific output heads
- Designed to run on **Kaggle GPU** ([view on Kaggle](https://www.kaggle.com/code/anjal12/nepal-stock-lstm-mtl))

## 🛠️ Requirements
```
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost
tensorflow / keras (for LSTM notebook)
```

---
*By Anjal Subedi | Data: NEPSE 2012–2020*
