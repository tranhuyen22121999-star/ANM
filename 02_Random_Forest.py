from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "Results"

BASELINE_TYPES = ["Benign", "DoS", "DDoS", "Portscan"]
NOVEL_TYPES = [
    "Botnet", "Bruteforce", "Infiltration", "WebAttacks"
]

# Những cột nhận diện hoặc mang thông tin nhãn không được dùng
# làm đặc trưng dự đoán.
EXCLUDED_COLUMNS = {
    "AttackType",
    "BinaryLabel",
    "Label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Src IP",
    "Dst IP",
    "Timestamp",
}

def main():
    path = RESULT_DIR / "cleaned_cic_ids2017.parquet"
    df = pd.read_parquet(path)

    # Chuẩn hóa tên cột.
    df.columns = df.columns.astype(str).str.strip()

    required = {"AttackType", "BinaryLabel"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    baseline = df[df["AttackType"].isin(BASELINE_TYPES)].copy()
    novel = df[df["AttackType"].isin(NOVEL_TYPES)].copy()

    if baseline.empty:
        raise ValueError("Baseline dataset is empty.")
    if novel.empty:
        raise ValueError("Novel attack dataset is empty.")

    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    # Loại cột có tên thường chứa định danh.
    # Cần xem xét thủ công nếu dataset có tên cột khác.
    feature_cols = [
        col for col in feature_cols
        if col.lower() not in {
            "flow_id", "source_ip", "destination_ip",
            "src_ip", "dst_ip", "timestamp", "label"
        }
    ]

    # Chỉ loại cột không có giá trị trong baseline.
    feature_cols = [
        col for col in feature_cols
        if not baseline[col].isna().all()
    ]

    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    # Loại bỏ các dòng không có nhãn nhị phân hợp lệ.
    baseline = baseline.dropna(subset=["BinaryLabel"])
    novel = novel.dropna(subset=["BinaryLabel"])

    # Chia theo dòng trong baseline; tập novel không tham gia train.
    train_df, test_df = train_test_split(
        baseline,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=baseline["BinaryLabel"],
    )

    # Lưu dữ liệu thô. Encoder/imputer sẽ fit ở bước huấn luyện.
    train_df.to_parquet(
        RESULT_DIR / "baseline_train.parquet", index=False
    )
    test_df.to_parquet(
        RESULT_DIR / "baseline_test.parquet", index=False
    )
    novel.to_parquet(
        RESULT_DIR / "novel_attacks.parquet", index=False
    )

    config = {
        "feature_columns": feature_cols,
        "baseline_types": BASELINE_TYPES,
        "novel_types": NOVEL_TYPES,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "novel_rows": int(len(novel)),
    }

    with open(
        RESULT_DIR / "feature_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Baseline train:", train_df.shape)
    print("Baseline test:", test_df.shape)
    print("Novel attacks:", novel.shape)
    print("\nTrain distribution:")
    print(train_df["BinaryLabel"].value_counts())
    print("\nNovel attack distribution:")
    print(novel["AttackType"].value_counts())
    print("\nSaved split datasets and feature configuration.")

if __name__ == "__main__":
    main()
