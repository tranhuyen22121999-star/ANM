from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "Results"

MAX_SAMPLE_PER_GROUP = 20000
RANDOM_STATE = 42
DRIFT_THRESHOLD = 0.20
ALPHA = 0.05

def main():
    train = pd.read_parquet(
        RESULT_DIR / "baseline_train.parquet"
    )
    novel = pd.read_parquet(
        RESULT_DIR / "novel_attacks.parquet"
    )

    with open(
        RESULT_DIR / "feature_config.json",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    features = config["feature_columns"]

    preprocessor = joblib.load(
        RESULT_DIR / "preprocessor.joblib"
    )

    # Chỉ so sánh đặc trưng; nhãn và tên attack family không đưa vào X.
    X_base = preprocessor.transform(train[features])
    X_novel = preprocessor.transform(novel[features])

    feature_names = preprocessor.get_feature_names_out()

    # Chuyển thành DataFrame để giữ tên cột.
    X_base = pd.DataFrame(
        X_base,
        columns=feature_names,
    )
    X_novel = pd.DataFrame(
        X_novel,
        columns=feature_names,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    def sample_frame(frame):
        if len(frame) > MAX_SAMPLE_PER_GROUP:
            indices = rng.choice(
                len(frame),
                size=MAX_SAMPLE_PER_GROUP,
                replace=False,
            )
            return frame.iloc[indices]
        return frame

    X_base = sample_frame(X_base)
    X_novel = sample_frame(X_novel)

    results = []

    for feature in feature_names:
        base_values = X_base[feature].to_numpy(dtype=float)
        novel_values = X_novel[feature].to_numpy(dtype=float)

        base_values = base_values[np.isfinite(base_values)]
        novel_values = novel_values[np.isfinite(novel_values)]

        if len(base_values) == 0 or len(novel_values) == 0:
            continue

        statistic, p_value = ks_2samp(
            base_values,
            novel_values,
            alternative="two-sided",
            method="auto",
        )

        results.append({
            "Feature": feature,
            "KS_Statistic": float(statistic),
            "P_Value": float(p_value),
            "Baseline_Mean": float(np.mean(base_values)),
            "Novel_Mean": float(np.mean(novel_values)),
            "Baseline_Std": float(np.std(base_values)),
            "Novel_Std": float(np.std(novel_values)),
        })

    drift_df = pd.DataFrame(results)

    if drift_df.empty:
        raise ValueError("No valid features for drift analysis.")

    # Hiệu chỉnh multiple testing.
    reject, adjusted_p, _, _ = multipletests(
        drift_df["P_Value"].to_numpy(),
        alpha=ALPHA,
        method="fdr_bh",
    )

    drift_df["Adjusted_P_Value"] = adjusted_p
    drift_df["Statistically_Significant"] = reject

    # KS statistic là khoảng cách cực đại giữa hai ECDF.
    # Đây là ngưỡng thực nghiệm, không phải ngưỡng phổ quát.
    drift_df["Drift_Flag"] = (
        (drift_df["KS_Statistic"] >= DRIFT_THRESHOLD)
        & drift_df["Statistically_Significant"]
    )

    drift_df = drift_df.sort_values(
        ["Drift_Flag", "KS_Statistic"],
        ascending=[False, False],
    )

    output_path = RESULT_DIR / "drift_results.csv"
    drift_df.to_csv(output_path, index=False)

    flagged = int(drift_df["Drift_Flag"].sum())
    total = int(len(drift_df))

    summary = {
        "comparison": "baseline training vs novel attack families",
        "baseline_rows_sampled": int(len(X_base)),
        "novel_rows_sampled": int(len(X_novel)),
        "features_tested": total,
        "features_flagged": flagged,
        "drift_ratio": float(flagged / total) if total else 0.0,
        "ks_threshold": DRIFT_THRESHOLD,
        "significance_level": ALPHA,
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "overall_drift_detected": bool(flagged > 0),
    }

    with open(
        RESULT_DIR / "drift_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nDrift Summary")
    print(json.dumps(summary, indent=2))

    print("\nTop 20 drifted features:")
    print(drift_df.head(20).to_string(index=False))

    print("\nSaved:", output_path)

if __name__ == "__main__":
    main()
