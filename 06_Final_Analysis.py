from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "Results"

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    rf_path = RESULT_DIR / "random_forest_pipeline.joblib"
    novel_path = RESULT_DIR / "novel_attacks.parquet"

    if not rf_path.exists():
        raise FileNotFoundError(
            "Run 03_Rung_Ngau_Nhien.py first."
        )

    if not novel_path.exists():
        raise FileNotFoundError(
            "Run 02_Ma_Hoa_Va_Trich_Chon_Dac_Trung.py first."
        )

    model = joblib.load(rf_path)
    novel = pd.read_parquet(novel_path)

    config = load_json(
        RESULT_DIR / "feature_config.json"
    )
    features = config["feature_columns"]

    X_novel = novel[features]
    y_novel = novel["BinaryLabel"].astype(int).to_numpy()

    # Đánh giá khả năng phát hiện các attack family không dùng để train.
    y_pred = model.predict(X_novel)

    novel_metrics = {
        "rows": int(len(novel)),
        "attack_families": {
            str(k): int(v)
            for k, v in novel["AttackType"].value_counts().items()
        },
        "accuracy": float(accuracy_score(y_novel, y_pred)),
        "precision": float(
            precision_score(y_novel, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_novel, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_novel, y_pred, zero_division=0)),
        "attack_detection_rate": float(np.mean(y_pred == 1)),
        "evaluation_warning": (
            "Novel dataset contains attack rows only. "
            "Specificity and false-positive rate cannot be evaluated "
            "on this subset."
        ),
    }

    novel_report = classification_report(
        y_novel,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
        output_dict=True,
    )

    with open(
        RESULT_DIR / "novel_attack_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(novel_metrics, f, indent=2)

    with open(
        RESULT_DIR / "novel_attack_classification_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(novel_report, f, indent=2)

    # Đọc các kết quả đã sinh ra từ notebook/script trước.
    rf_metrics = load_json(
        RESULT_DIR / "random_forest_metrics.json"
    )
    iso_metrics = load_json(
        RESULT_DIR / "isolation_forest_metrics.json"
    )
    drift_summary = load_json(
        RESULT_DIR / "drift_summary.json"
    )

    # Cảnh báo nếu F1 trên tập novel giảm mạnh so với baseline.
    baseline_f1 = rf_metrics["f1"]
    novel_f1 = novel_metrics["f1"]

    f1_drop = baseline_f1 - novel_f1

    # Ngưỡng vận hành ban đầu; cần hiệu chỉnh bằng validation data.
    PERFORMANCE_DROP_THRESHOLD = 0.10

    performance_alert = bool(
        f1_drop >= PERFORMANCE_DROP_THRESHOLD
    )

    drift_alert = bool(
        drift_summary["overall_drift_detected"]
    )

    monitoring_status = (
        "REVIEW_REQUIRED"
        if performance_alert or drift_alert
        else "NO_ALERT_FROM_CONFIGURED_RULES"
    )

    monitoring_summary = {
        "project": "CIC-IDS2017 Network Anomaly Detection",
        "random_forest_baseline": rf_metrics,
        "isolation_forest_baseline": iso_metrics,
        "novel_attack_evaluation": novel_metrics,
        "baseline_f1": float(baseline_f1),
        "novel_attack_f1": float(novel_f1),
        "f1_drop": float(f1_drop),
        "performance_drop_threshold": PERFORMANCE_DROP_THRESHOLD,
        "performance_alert": performance_alert,
        "drift_summary": drift_summary,
        "drift_alert": drift_alert,
        "monitoring_status": monitoring_status,
        "human_in_the_loop": (
            "Analyst should review alerts, investigate suspicious "
            "traffic, verify labels and decide whether retraining "
            "is justified."
        ),
        "limitations": [
            "Novel attack evaluation contains attack rows only.",
            "Distribution drift does not prove model degradation.",
            "Random train/test splitting may allow correlated flows "
            "from the same capture to appear in both partitions.",
            "Operational thresholds require validation on representative "
            "traffic and benign data.",
        ],
    }

    with open(
        RESULT_DIR / "monitoring_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(monitoring_summary, f, indent=2)

    comparison = pd.DataFrame([
        {
            "Model": "Random Forest",
            "Dataset": "Baseline test",
            "Accuracy": rf_metrics["accuracy"],
            "Precision": rf_metrics["precision"],
            "Recall": rf_metrics["recall"],
            "F1": rf_metrics["f1"],
            "FPR": rf_metrics["fpr"],
        },
        {
            "Model": "Isolation Forest",
            "Dataset": "Baseline test",
            "Accuracy": iso_metrics["accuracy"],
            "Precision": iso_metrics["precision"],
            "Recall": iso_metrics["recall"],
            "F1": iso_metrics["f1"],
            "FPR": iso_metrics["fpr"],
        },
        {
            "Model": "Random Forest",
            "Dataset": "Novel attacks only",
            "Accuracy": novel_metrics["accuracy"],
            "Precision": novel_metrics["precision"],
            "Recall": novel_metrics["recall"],
            "F1": novel_metrics["f1"],
            "FPR": np.nan,
        },
    ])

    comparison.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n=== NOVEL ATTACK EVALUATION ===")
    print(json.dumps(novel_metrics, indent=2))

    print("\n=== MODEL COMPARISON ===")
    print(comparison.to_string(index=False))

    print("\n=== MONITORING STATUS ===")
    print(monitoring_status)

    print("\n=== HUMAN REVIEW ===")
    print(monitoring_summary["human_in_the_loop"])

    print("\nSaved monitoring_summary.json and model_comparison.csv")

if __name__ == "__main__":
    main()
