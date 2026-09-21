# joblib: trained model ko file mein save karne ke liye
import joblib
import pandas as pd
from pathlib import Path

# sklearn ke tools: split, preprocessing, pipeline, models, metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

# Step 3 mein banaya hua function
from features import add_features

CLEAN_PATH = "data/processed/telco_clean.csv"
MODEL_PATH = "models/model.joblib"
TARGET = "Churn"

# Number wale columns: inko StandardScaler se same scale pe layenge
NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_addon_services", "is_month_to_month", "is_auto_pay", "avg_monthly_charge",
]

# Text wale columns: inko OneHotEncoder se 0/1 columns mein badlenge
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def build_pipeline(model):
    """Preprocessing + model ko ek Pipeline mein jodo."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        # handle_unknown="ignore": API mein koi nayi category aayi toh crash nahi hoga
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test):
    """Model ko test data pe check karo, metrics dict wapas do."""
    y_pred = pipeline.predict(X_test)                    # 0 ya 1
    y_proba = pipeline.predict_proba(X_test)[:, 1]       # churn ki probability

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n=== {name} ===")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    return metrics


def main():
    # 1. Data padho aur features jodo
    df = add_features(pd.read_csv(CLEAN_PATH))

    # 2. X (input columns) aur y (jo predict karna hai) alag karo
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    # 3. Train/test split: 80% seekhne ke liye, 20% imtihaan ke liye
    # stratify=y: dono hisso mein churn ka ratio same rahe
    # random_state=42: har baar same split mile
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print(f"Baseline (sabko 'No' bolo) accuracy: {1 - y_test.mean():.3f}")

    # 4. Do models jinko compare karenge
    # class_weight="balanced": churn wale customers ko zyada importance do (imbalance ka ilaaj)
    candidates = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    # 5. Har model train karo aur evaluate karo
    results, pipelines = {}, {}
    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)          # yahan model seekhta hai
        results[name] = evaluate(name, pipeline, X_test, y_test)
        pipelines[name] = pipeline

    # 6. Jiska ROC-AUC zyada ho, wo winner
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nWinner: {best_name}")

    # 7. Winner ko save karo
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipelines[best_name], MODEL_PATH)
    print("Saved:", MODEL_PATH)

    # 8. Save hua model wapas load karke ek customer pe test karo
    loaded = joblib.load(MODEL_PATH)
    sample = X_test.iloc[[0]]
    proba = loaded.predict_proba(sample)[0, 1]
    print(f"\nReload test -> churn probability: {proba:.2f} | asli jawab: {y_test.iloc[0]}")


if __name__ == "__main__":
    main()