# NBA-Playoffs-Qualificaton-Forecaster

# Predicting NBA Playoff Qualification from Early-Season Performance – ML Classification

**TL;DR:** I built a machine learning system to predict if an NBA team will make the playoffs using only the first 40% of their season. Using an SMO (SVM) model and a custom-built Schedule Strength feature, I achieved 79.57% accuracy across 73 years of historical data, proving that early-season performance can be a reliable predictor of final outcomes.

---

## Project Overview
I created this project to investigate "temporal generalization" in sports—specifically, whether I can accurately predict a team's end-of-season playoff status using only a snapshot of their first **40% of games**. This is a practical challenge because teams and fans (myself included as a worried Warriors fan) want actionable signals long before the 82nd game is played.

By restricting my features to this early window, I enforced a realistic constraint that mirrors real-world prediction scenarios. I utilized a historical dataset spanning from 1947 to 2020 to ensure my model could generalize across different eras of basketball.

## Quick Summary
* **Peak Accuracy:** Predicted playoff qualification with a peak accuracy of **80.91%** using a Support Vector Machine (SMO) on held-out test sets.
* **Baseline Comparison:** My models significantly outpaced a majority-class baseline of approximately 60%.
* **Core Drivers:** The most influential features for prediction were overall win percentage and my custom schedule strength metric.
* **Open Source:** I have shared the final runnable model as `smoFinal.model` so others can test these predictions on modern data.

## Data Collection & Feature Engineering

### Data Retrieval
I extracted 63,157 game-level records from `RAW-NBA-Game-Results.csv`, which contains historical logs from 1947–2020.

### Key Predictive Metrics
For each team-season, I calculated eight key metrics based strictly on the first 40% window:
* **WinPct**: Overall team quality.
* **HomeWinPct**: Performance at home.
* **BlowoutWinPct**: Percentage of wins by 15+ points (dominance measure).
* **AvgPointDiff**: Net rating proxy.
* **SdPointDiff**: Consistency and volatility indicator.
* **PPG**: Pure offensive output (Points Per Game).
* **Last5WinPct**: Recent momentum within the early window.
* **AvgOppWinPct**: Schedule strength indicator.

### Schedule Strength Logic
To fix errors where the model was fooled by teams beating weak opponents, I implemented **AvgOppWinPct**. I calculated each opponent's win percentage using only *their* first 40% of games to ensure no future data leaked into my current prediction.

### Era-Block Partitioning
I grouped seasons into 5-year blocks to prevent the model from learning era-specific artifacts like rule changes or expansion dynamics. This ensures the model learns "basketball strength" rather than just the trends of a specific decade.

## Machine Learning Models & Performance

| Model | Accuracy (%) | Kappa |
| :--- | :--- | :--- |
| **SVM (SMO) - Final** | **79.57** | **0.570** |
| J48 Decision Tree | 80.38 | 0.576 |
| Logistic Regression | 80.11 | 0.586 |
| Random Forest | 79.03 | 0.560 |

I ultimately chose the **SMO (SVM)** because its margin-based approach provided the best balance between flexibility and generalization on these specific features. I tuned the complexity parameter to `c=1.0` to achieve the best stability across my test sets.

## Files & Structure
* **`Rcode.R`**: My full script for data normalization, date parsing, and feature engineering.
* **`smoFinal.model`**: The final trained Weka model ready for deployment.
* **`TRAIN_feature_data.csv`**: My training set of 372 team-seasons.
* **`TEST_feature_data.csv`**: The final held-out evaluation set of 93 team-seasons.
* **`DEV_feature_data.csv`**: Data used for parameter tuning.
* **`FullWriteup.md`**: My detailed project report.

## Key Results & Insights
* **The Winning Feature**: Adding the schedule strength metric alone improved my accuracy by over 2% and directly fixed many false positive errors.
* **The Performance Ceiling**: I noticed that all models plateaued around 80%. This tells me that ~20% of playoff qualification is determined by unpredictable factors like mid-season injuries or trades that aren't visible in the first 40% of the season.
* **Critical Predictors**: Win percentage is the single most important signal; removing it caused my accuracy to drop by over 8%.

---
