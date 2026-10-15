from pathlib import Path
import json
import gc
import numpy as np
import pandas as pd

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Dataset Full"
RESULT_DIR = BASE_DIR / "Results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Benign": "Benign-Monday-no-metadata.parquet",
    "Botnet": "Botnet-Friday-no-metadata.parquet",
    "Bruteforce": "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS": "DDoS-Friday-no-metadata.parquet",
    "DoS": "DoS-Wednesday-no-metadata.parquet",
    "Infiltration": "Infiltration-Thursday-no-metadata.parquet",
    "Portscan": "Portscan-Friday-no-metadata.parquet",
    "WebAttacks": "WebAttacks-Thursday-no-metadata.parquet",
}

def main():
    frames = []

    for attack_type, filename in FILES.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        part = pd.read_parquet(path)
        part.columns = part.columns.astype(str).str.strip()

        # Lưu nguồn dữ liệu để truy vết và phân tích drift.
        part["AttackType"] = attack_type
        part["BinaryLabel"] = int(attack_type != "Benign")

        frames.append(part)

        print(
            f"{filename}: "
            f"{len(part):,} rows, {len(part.columns)} columns"
        )

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # Chuẩn hóa giá trị vô cực.
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    # Chỉ loại bỏ các dòng thiếu nhãn.
    df = df.dropna(subset=["AttackType", "BinaryLabel"])

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before - len(df)

    # Loại bỏ cột đặc trưng hoàn toàn rỗng.
    metadata_cols = {"AttackType", "BinaryLabel"}
    empty_features = [
        col for col in df.columns
        if col not in metadata_cols and df[col].isna().all()
    ]
    df = df.drop(columns=empty_features)

    # Không xóa toàn bộ dòng chỉ vì một vài đặc trưng bị thiếu.
    # Việc điền giá trị thiếu sẽ được học từ train ở bước 02/03.
    
    output_path = RESULT_DIR / "cleaned_cic_ids2017.parquet"
    df.to_parquet(output_path, index=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates_removed": int(removed_duplicates),
        "all_missing_columns_removed": empty_features,
        "attack_distribution": {
            str(k): int(v)
            for k, v in df["AttackType"].value_counts().items()
        },
        "binary_distribution": {
            str(k): int(v)
            for k, v in df["BinaryLabel"].value_counts().items()
        },
    }

    with open(
        RESULT_DIR / "data_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDataset shape:", df.shape)
    print("\nAttack distribution:")
    print(df["AttackType"].value_counts())
    print("\nSaved:", output_path)

if __name__ == "__main__":
    main()
