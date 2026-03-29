"""
从 scienceqa_data_fixed.csv 中筛掉本地无图的题目，并导出与 we_math_cleaned.csv 同列名的
scienceqa_cleaned.csv，便于直接复用 build_vectors.py。
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_CSV = os.path.join(BASE_DIR, "scienceqa_data_fixed.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "scienceqa_images_fixed")
OUT_CSV = os.path.join(BASE_DIR, "scienceqa_cleaned.csv")


def _cell(row, key: str) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _desc(row) -> str:
    lec = _cell(row, "lecture")
    exp = _cell(row, "explanation")
    if lec and exp:
        return (lec + "\n\n" + exp)[:8000]
    return (lec or exp)[:8000]


def main():
    if not os.path.isfile(SRC_CSV):
        raise FileNotFoundError(SRC_CSV)
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(IMAGE_DIR)

    df = pd.read_csv(SRC_CSV, encoding="utf-8", low_memory=False)
    n0 = len(df)

    keep_mask = []
    for _, row in df.iterrows():
        fn = str(row.get("image_filename", "") or "").strip()
        if not fn:
            keep_mask.append(False)
            continue
        full = os.path.join(IMAGE_DIR, os.path.basename(fn))
        keep_mask.append(os.path.isfile(full))

    sub = df.loc[keep_mask].copy()
    sub.reset_index(drop=True, inplace=True)

    out = pd.DataFrame(
        {
            "ID": sub["id"].astype(str),
            "question": sub["question"].astype(str),
            "knowledge concept": sub["category"].fillna("").astype(str),
            "option": sub["choices"].fillna("").astype(str),
            "answer": sub["answer_text"].fillna("").astype(str),
            "image_path": sub["image_filename"].apply(
                lambda x: f"scienceqa_images_fixed/{os.path.basename(str(x).strip())}"
            ),
            "knowledge concept description": sub.apply(_desc, axis=1),
        }
    )

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"源行数: {n0}，保留有图: {len(out)}，输出: {OUT_CSV}")


if __name__ == "__main__":
    main()
