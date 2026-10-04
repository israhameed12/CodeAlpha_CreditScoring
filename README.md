# CodeAlpha Task 1 — Credit Scoring Model

## Objective
Classify whether a record is labelled creditworthy (1) or not creditworthy (0) using example financial attributes.

## Dataset notice
`credit_data.csv` is a **synthetic, computer-generated teaching dataset** (1,200 rows). Its labels were generated from a simple artificial scoring rule with randomness. It is included so the project can be run immediately; it is not actual bank/customer data and cannot validate real lending decisions. Replace it with a properly licensed, documented dataset if CodeAlpha or your instructor requires real-world data.

## Features
- age
- annual_income
- existing_debt
- late_payments_last_2y
- employment_years
- loan_amount
- employment_type

Target: `creditworthy` (1/0).

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
```
The script performs an 80/20 stratified train/test split, imputes missing values, scales numeric fields, one-hot encodes categories, compares Logistic Regression, Decision Tree and Random Forest, and prints Accuracy, Precision, Recall, F1, ROC-AUC, a classification report and confusion matrix. It saves the selected model and `metrics.json` under `outputs/`.

## Important
The chosen model is selected by test F1 for a classroom comparison only. For an unbiased final evaluation, use a separate validation set or nested/cross-validation during model selection. This is a learning exercise, not a real credit approval tool. Discuss fairness, privacy, class imbalance and false-positive/false-negative costs in your report.
