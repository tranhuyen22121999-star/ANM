from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
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

def main():
    train = pd.read_parquet(
        RESULT_DIR / "baseline_train.parquet"
    )
    test = pd.read_parquet(
        RESULT_DIR / "baseline_test.parquet"
    )

    preprocessor = joblib.load(
        RESULT_DIR / "preprocessor.joblib"
    )

    with open(
        RESULT_DIR / "feature_config.json",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    features = config["feature_columns"]

    # Isolation Forest chỉ học trên benign training flows.
    benign_train = train[train["BinaryLabel"] == 0]

    if benign_train.empty:
        raise ValueError("No benign training rows found.")

    X_benign = preprocessor.transform(
        benign_train[features]
    )
    X_test = preprocessor.transform(test[features])

    y_test = test["BinaryLabel"].astype(int).to_numpy()

    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    detector.fit(X_benign)

    # IsolationForest: 1 = normal, -1 = anomaly.
    raw_predictions = detector.predict(X_test)

    # Chuẩn hóa nhãn: 0 = benign, 1 = attack.
    y_pred = np.where(raw_predictions == -1, 1, 0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "benign_training_rows": int(len(benign_train)),
    }

    print("\nIsolation Forest Metrics")
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
    plt.title("Isolation Forest - Baseline Test")
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "isolation_forest_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    joblib.dump(
        {
            "detector": detector,
            "preprocessor": preprocessor,
            "feature_columns": features,
        },
        RESULT_DIR / "isolation_forest_bundle.joblib",
    )

    with open(
        RESULT_DIR / "isolation_forest_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved Isolation Forest model and metrics.")

if __name__ == "__main__":
    main()