from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "Results"

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def main():
    with open(
        RESULT_DIR / "feature_config.json",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    features = config["feature_columns"]

    train = pd.read_parquet(
        RESULT_DIR / "baseline_train.parquet"
    )
    test = pd.read_parquet(
        RESULT_DIR / "baseline_test.parquet"
    )

    X_train = train[features].copy()
    y_train = train["BinaryLabel"].astype(int)

    X_test = test[features].copy()
    y_test = test["BinaryLabel"].astype(int)

    numeric_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = [
        col for col in features
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print("\nRandom Forest Metrics")
    print(json.dumps(metrics, indent=2))

    print("\nClassification Report")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["Benign", "Attack"],
        values_format="d",
    )
    plt.title("Random Forest - Baseline Test")
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "random_forest_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # Lưu mô hình có cả bước tiền xử lý.
    joblib.dump(model, RESULT_DIR / "random_forest_pipeline.joblib")

    # Lưu riêng bộ tiền xử lý để dùng cho Isolation Forest và drift.
    joblib.dump(
        model.named_steps["preprocessor"],
        RESULT_DIR / "preprocessor.joblib",
    )

    with open(
        RESULT_DIR / "random_forest_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    # Feature importance.
    importances = model.named_steps["classifier"].feature_importances_
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        RESULT_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    top = importance_df.head(20).sort_values("Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top["Feature"], top["Importance"])
    plt.title("Top 20 Random Forest Features")
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "random_forest_feature_importance.png",
        dpi=200,
    )
    plt.close()

    print("\nSaved model, metrics, confusion matrix and feature importance.")

if __name__ == "__main__":
    main()
