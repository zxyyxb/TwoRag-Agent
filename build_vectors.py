"""
离线数据构建层：文本向量化 + 图像向量化
使用 NumPy 向量存储（避免 ChromaDB HNSW 在 Windows 上的兼容问题），共用 ID 实现跨模态映射
"""
import os

# 国内网络环境下使用 HuggingFace 镜像（在 import 模型前设置）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import open_clip
import torch
from tqdm import tqdm

from config import (
    CSV_PATH,
    IMAGE_BASE_DIR,
    CHROMA_PERSIST_DIR,
    TEXT_EMBEDDING_MODEL,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
)
from vector_store import NumpyVectorStore


def load_dataset(csv_path: str) -> pd.DataFrame:
    """加载 CSV 数据集，正确处理多行字段（knowledge concept description 等可能含换行）"""
    # 标准 CSV，逗号分隔，引号内可含换行
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, sep="\t", encoding="utf-8", low_memory=False)
    return df


def build_text_for_embedding(row) -> str:
    """构建用于文本向量化的文本：题目 + 知识点"""
    question = str(row.get("question", "") or "")
    knowledge = str(row.get("knowledge concept", "") or "")
    return f"{question}\n{knowledge}".strip()


def get_image_full_path(image_path: str) -> str:
    """根据 CSV 中的 image_path 获取实际图像完整路径"""
    # image_path 可能是 extracted_images/1-1.png
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(IMAGE_BASE_DIR, os.path.basename(image_path))


def main():
    print("=" * 50)
    print("离线数据构建：文本向量 + 图像向量")
    print("=" * 50)

    # 1. 加载数据
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 不存在: {CSV_PATH}")
    df = load_dataset(CSV_PATH)
    print(f"加载数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    # 2. 初始化向量存储（覆盖旧数据）
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
    store = NumpyVectorStore(CHROMA_PERSIST_DIR)

    # 3. 加载文本嵌入模型
    print("加载文本嵌入模型...")
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

    # 4. 加载 CLIP 图像模型
    print("加载 CLIP 图像模型...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    # 5. 准备批量数据
    ids = []
    texts = []
    metadatas = []
    valid_indices = []  # 记录有效行索引，用于图像

    for idx, row in df.iterrows():
        # ChromaDB 要求 ID 唯一，CSV 中可能存在重复 ID，故用行下标保证唯一
        rid = f"row_{idx}"
        ids.append(rid)

        text = build_text_for_embedding(row)
        texts.append(text if text else " ")

        md = {
            "original_id": str(row.get("ID", idx)),
            "question": str(row.get("question", ""))[:1000],
            "knowledge_concept": str(row.get("knowledge concept", ""))[:500],
            "answer": str(row.get("answer", "")),
            "image_path": str(row.get("image_path", "")),
            "option": str(row.get("option", ""))[:500],
            "knowledge_concept_desc": str(row.get("knowledge concept description", ""))[:2000],
        }
        metadatas.append(md)
        valid_indices.append(idx)

    # 6. 文本向量化并入库
    print("文本向量化...")
    text_embeddings = text_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    store.save_text_collection(ids, text_embeddings, metadatas, texts)
    print(f"文本向量集合已构建，共 {len(ids)} 条")

    # 7. 图像向量化并入库
    print("图像向量化...")
    image_ids = []
    image_embeddings_list = []
    image_metadatas = []
    skipped = 0

    for i, idx in enumerate(tqdm(valid_indices, desc="处理图像")):
        row = df.iloc[idx]
        img_path = row.get("image_path", "")
        full_path = get_image_full_path(img_path)

        if not os.path.exists(full_path):
            # 尝试直接用 image_path
            alt = os.path.join(IMAGE_BASE_DIR, img_path) if img_path else ""
            if os.path.exists(alt):
                full_path = alt
            else:
                skipped += 1
                continue

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"跳过图像 {full_path}: {e}")
            skipped += 1
            continue

        with torch.no_grad():
            img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
            emb = clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            image_embeddings_list.append(emb.cpu().numpy().flatten().tolist())

        image_ids.append(ids[i])
        image_metadatas.append(metadatas[i])

    if skipped > 0:
        print(f"跳过 {skipped} 张无法加载的图像")

    # 写入图像向量集合
    img_emb = np.array(image_embeddings_list, dtype=np.float32)
    store.save_image_collection(image_ids, img_emb, image_metadatas)
    print(f"图像向量集合已构建，共 {len(image_ids)} 条")

    print("=" * 50)
    print("离线构建完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
