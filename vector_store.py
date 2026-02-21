"""
基于 NumPy 的轻量向量存储，替代 ChromaDB 避免 Windows 下 HNSW 兼容性问题
使用余弦相似度暴力检索，数据量约 2k 时毫秒级响应
"""
import os
import json
import numpy as np


class NumpyVectorStore:
    """NumPy 向量存储，持久化到本地文件"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.text_ids = []
        self.text_embeddings = np.array([])
        self.text_metadatas = []
        self.text_documents = []
        self.image_ids = []
        self.image_embeddings = np.array([])
        self.image_metadatas = []
        self._id_to_img_idx = {}

    def save_text_collection(self, ids: list, embeddings: np.ndarray, metadatas: list, documents: list):
        """保存文本向量集合"""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.text_ids = ids
        self.text_embeddings = np.array(embeddings, dtype=np.float32)
        self.text_metadatas = metadatas
        self.text_documents = documents
        np.save(os.path.join(self.persist_dir, "text_embeddings.npy"), self.text_embeddings)
        with open(os.path.join(self.persist_dir, "text_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "metadatas": metadatas, "documents": documents}, f, ensure_ascii=False, indent=0)

    def save_image_collection(self, ids: list, embeddings: np.ndarray, metadatas: list):
        """保存图像向量集合"""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.image_ids = ids
        self.image_embeddings = np.array(embeddings, dtype=np.float32)
        self.image_metadatas = metadatas
        self._id_to_img_idx = {id_: i for i, id_ in enumerate(ids)}
        np.save(os.path.join(self.persist_dir, "image_embeddings.npy"), self.image_embeddings)
        with open(os.path.join(self.persist_dir, "image_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "metadatas": metadatas}, f, ensure_ascii=False, indent=0)

    def load(self):
        """从磁盘加载"""
        text_emb_path = os.path.join(self.persist_dir, "text_embeddings.npy")
        text_meta_path = os.path.join(self.persist_dir, "text_meta.json")
        img_emb_path = os.path.join(self.persist_dir, "image_embeddings.npy")
        img_meta_path = os.path.join(self.persist_dir, "image_meta.json")

        if not os.path.exists(text_emb_path) or not os.path.exists(text_meta_path):
            raise FileNotFoundError(f"向量库不存在，请先运行 build_vectors.py 构建。目录: {self.persist_dir}")

        self.text_embeddings = np.load(text_emb_path)
        with open(text_meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.text_ids = data["ids"]
        self.text_metadatas = data["metadatas"]
        self.text_documents = data.get("documents", [""] * len(self.text_ids))

        if os.path.exists(img_emb_path) and os.path.exists(img_meta_path):
            self.image_embeddings = np.load(img_emb_path)
            with open(img_meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.image_ids = data["ids"]
            self.image_metadatas = data["metadatas"]
            self._id_to_img_idx = {id_: i for i, id_ in enumerate(self.image_ids)}
        else:
            self.image_ids = []
            self.image_embeddings = np.array([])
            self.image_metadatas = []
            self._id_to_img_idx = {}

    def text_query(self, query_embedding: np.ndarray, n_results: int) -> dict:
        """
        文本相似度检索，返回 ChromaDB 兼容格式
        query_embedding: (D,) 或 (1, D)，已归一化
        """
        if self.text_embeddings.size == 0:
            return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.text_embeddings.shape[1]:
            return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        # 余弦相似度（已归一化时等价于点积）
        sim = q @ self.text_embeddings.T
        sim = sim.ravel()
        # 距离取 1 - 相似度（越小越相似）
        dist = 1.0 - sim
        top_idx = np.argsort(dist)[:n_results]
        return {
            "ids": [[self.text_ids[i] for i in top_idx]],
            "metadatas": [[self.text_metadatas[i] for i in top_idx]],
            "documents": [[self.text_documents[i] for i in top_idx]],
            "distances": [dist[top_idx].tolist()],
        }

    def image_get(self, ids: list, include: list | None = None) -> dict:
        """
        根据 ID 获取图像向量，ChromaDB 兼容格式
        """
        include = include or ["embeddings", "metadatas"]
        result_ids = []
        result_embeddings = []
        result_metadatas = []
        for id_ in ids:
            if id_ in self._id_to_img_idx:
                idx = self._id_to_img_idx[id_]
                result_ids.append(id_)
                if "embeddings" in include:
                    result_embeddings.append(self.image_embeddings[idx].tolist())
                if "metadatas" in include:
                    result_metadatas.append(self.image_metadatas[idx])
        return {
            "ids": result_ids,
            "embeddings": result_embeddings if "embeddings" in include else None,
            "metadatas": result_metadatas if "metadatas" in include else None,
        }

    def text_count(self) -> int:
        return len(self.text_ids)
