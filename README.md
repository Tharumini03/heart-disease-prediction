# Heart Disease Prediction

This repository contains my solution for the **Kaggle Playground Series S6E2 — Heart Disease Prediction** competition.

## Kaggle Information

- **Kaggle Username:** UOM_230194C
- **Competition:** Playground Series S6E2 — Heart Disease Prediction
- **Evaluation Metric:** AUC-ROC
- **Final Public Score:** 0.95301

## Project Overview

The goal of this project was to predict the probability of heart disease using structured clinical data. The problem was treated as a **binary classification** task, where the target variable had two classes:

- `Absence` → 0
- `Presence` → 1

The final model was trained using **HistGradientBoostingClassifier** from scikit-learn.

## Dataset Overview

The dataset contained:

- **630,000** training samples
- **270,000** test samples
- **13** clinical input features

The target variable was **Heart Disease**.

## Preprocessing Steps

The following preprocessing steps were applied:

- Loaded `train.csv` and `test.csv` using pandas
- Converted the target variable from categorical to binary
- Removed the `id` column from model input features
- Used stratified train-validation split to preserve class balance
- Applied feature scaling only for the Logistic Regression baseline
- No missing value imputation was required

## Models Used

### 1. Logistic Regression (Baseline)
A Logistic Regression model was used as a baseline with `StandardScaler` inside a pipeline.

### 2. HistGradientBoostingClassifier (Final Model)
The final model used was `HistGradientBoostingClassifier` with the following hyperparameters:

- `learning_rate = 0.05`
- `max_depth = 6`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 30`
- `max_iter = 300`
- `random_state = 42`

## Validation Strategy

- 80/20 stratified train-validation split
- 5-Fold Stratified Cross-Validation

## Results

| Model | AUC |
|------|------|
| Logistic Regression (Baseline) | ~0.910 |
| HistGradientBoosting (Validation) | 0.9547 |
| HistGradientBoosting (5-Fold CV) | ~0.9548 |
| Final Kaggle Public Score | 0.95301 |

## Files in This Repository

- `main.py` — main Python code for training and prediction
- `submission.csv` — final competition submission file
- `README.md` — project documentation

## How to Run

1. Place `train.csv` and `test.csv` in the project folder
2. Install required libraries:

```bash
pip install pandas numpy scikit-learn
```
3. Run:

```bash
python main.py
```

## Conclusion
This project shows that gradient boosting methods can perform very well on structured medical datasets with minimal preprocessing. The final model achieved a public leaderboard score of 0.95301 AUC-ROC.

## Author
UOM_230194C