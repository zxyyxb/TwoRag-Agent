"""
将 WeMath 与 ScienceQA 清洗表合并为单一 merged_rag_corpus.csv。
image_path 统一为相对项目根：extracted_images/... 或 scienceqa_images_fixed/...，
供 build_vectors / RAG 双目录解析。
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WE_MATH = os.path.join(BASE_DIR, "we_math_cleaned.csv")
SCIENCEQA = os.path.join(BASE_DIR, "scienceqa_cleaned.csv")
OUT = os.path.join(BASE_DIR, "merged_rag_corpus.csv")

COLS = [
    "ID",
    "question",
    "knowledge concept",
    "option",
    "answer",
    "image_path",
    "knowledge concept description",
]


def _norm(subdir: str, image_path: str) -> str:
    p = str(image_path or "").strip().replace("\\", "/")
    if not p:
        return p
    if p.startswith(subdir + "/"):
        return p
    if "/" in p:
        return p
    return f"{subdir}/{os.path.basename(p)}"


def main():
    if not os.path.isfile(WE_MATH):
        raise FileNotFoundError(WE_MATH)
    if not os.path.isfile(SCIENCEQA):
        raise FileNotFoundError(SCIENCEQA)

    wm = pd.read_csv(WE_MATH, encoding="utf-8", low_memory=False)
    sq = pd.read_csv(SCIENCEQA, encoding="utf-8", low_memory=False)

    wm = wm.reindex(columns=COLS)
    sq = sq.reindex(columns=COLS)

    wm = wm.copy()
    sq = sq.copy()
    wm["image_path"] = wm["image_path"].apply(lambda x: _norm("extracted_images", x))
    sq["image_path"] = sq["image_path"].apply(lambda x: _norm("scienceqa_images_fixed", x))

    merged = pd.concat([wm, sq], ignore_index=True)
    merged.to_csv(OUT, index=False, encoding="utf-8")
    print(f"WeMath: {len(wm)} 条，ScienceQA: {len(sq)} 条，合并: {len(merged)} 条 → {OUT}")


if __name__ == "__main__":
    main()
