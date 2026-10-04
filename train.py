import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    p = argparse.ArgumentParser(description="Educational binary credit scoring classifier")
    p.add_argument("--data", default="credit_data.csv")
    p.add_argument("--target", default="creditworthy")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    df = pd.read_csv(args.data).drop_duplicates().dropna(subset=[args.target])
    X, y = df.drop(columns=[args.target]), df[args.target]
    if y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    results, fitted = {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4)
        }
        results[name] = metrics
        fitted[name] = pipe
        print(f"\\n{name}: {metrics}")
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix:\\n", confusion_matrix(y_test, pred))
    best = max(results, key=lambda k: results[k]["f1_score"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "credit_scoring_model.joblib")
    (out / "metrics.json").write_text(json.dumps({"model_selected_by": "highest test F1 (educational comparison)", "selected_model": best, "results": results}, indent=2))
    print(f"\\nSelected model: {best}")
    print(f"Saved model and metrics to: {out.resolve()}")

if __name__ == "__main__":
    main()
