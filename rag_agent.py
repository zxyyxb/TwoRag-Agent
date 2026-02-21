"""
两阶段多模态 RAG Agent
流程：文本 RAG 粗筛 → 靶向词生成 → 图像 RAG 精筛 → 聚合回答
"""
import os

# 国内网络环境下使用 HuggingFace 镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import open_clip
import torch

from config import (
    IMAGE_BASE_DIR,
    CHROMA_PERSIST_DIR,
    TEXT_TOP_K,
    IMAGE_TOP_K,
    IMAGE_SIM_WEIGHT,
    KEYWORD_SIM_WEIGHT,
    TEXT_EMBEDDING_MODEL,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
)
from vector_store import NumpyVectorStore
from answer_generator import generate_answer


def get_image_full_path(image_path: str) -> str:
    if not image_path:
        return ""
    if os.path.isabs(image_path):
        return image_path
    base = os.path.basename(image_path)
    return os.path.join(IMAGE_BASE_DIR, base)


class RAGAgent:
    """两阶段多模态 RAG Agent"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._store = None

    def _load_models(self):
        if self._text_model is None:
            print("加载文本嵌入模型...")
            self._text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        if self._clip_model is None:
            print("加载 CLIP 模型...")
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
            )
            self._clip_model.eval()
            self._clip_model = self._clip_model.to(self.device)

    def _load_store(self):
        if self._store is None:
            self._store = NumpyVectorStore(CHROMA_PERSIST_DIR)
            self._store.load()

    # ---------- 第一阶段：文本 RAG 粗检索 ----------
    def text_rag_retrieve(self, user_question: str, top_k: int = TEXT_TOP_K) -> list[dict]:
        """文本 RAG 粗筛，返回 Top-K 候选"""
        self._load_models()
        self._load_store()

        q_emb = self._text_model.encode([user_question], normalize_embeddings=True)
        n_results = min(top_k, self._store.text_count())
        results = self._store.text_query(q_emb, n_results=n_results)

        candidates = []
        if results["ids"] and results["ids"][0]:
            for i, rid in enumerate(results["ids"][0]):
                candidates.append({
                    "id": rid,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
        return candidates

    # ---------- 中间环节：靶向词生成 ----------
    def generate_targeted_keywords(self, candidates: list[dict], max_keywords: int = 10) -> list[str]:
        """
        基于文本 RAG 候选数据，提炼靶向词（关键词/主题/概念）
        用于辅助图像 RAG 做更精准的语义对齐
        """
        # 聚合候选集中的题目、知识点、答案
        all_texts = []
        for c in candidates[:min(15, len(candidates))]:
            md = c.get("metadata", {})
            doc = c.get("document", "")
            all_texts.append(doc or "")
            all_texts.append(md.get("knowledge_concept", ""))
            all_texts.append(md.get("question", ""))
        combined = " ".join(filter(None, all_texts))

        # 简单规则：提取英文术语、数学符号、几何图形相关词
        keywords = set()

        # 1. 几何/数学常见词（英文）
        math_terms = [
            "square", "circle", "triangle", "trapezoid", "cylinder", "cone",
            "radius", "diameter", "circumference", "arc", "sector", "angle",
            "volume", "height", "base", "perimeter", "rotation", "parallelogram",
        ]
        combined_lower = combined.lower()
        for t in math_terms:
            if t in combined_lower:
                keywords.add(t)

        # 2. 从 knowledge_concept 提取（按分号/逗号分割）
        for c in candidates[:10]:
            kc = c.get("metadata", {}).get("knowledge_concept", "")
            for part in re.split(r"[;,\n]", kc):
                part = part.strip()
                if 3 <= len(part) <= 40 and part.isascii():
                    keywords.add(part)
                # 取前几个词作为短语
                words = part.split()
                if len(words) >= 2:
                    keywords.add(" ".join(words[:2]))

        # 3. 中文关键概念映射（常见几何）
        cn_map = {
            "正方形": "square", "圆": "circle", "梯形": "trapezoid",
            "圆柱": "cylinder", "圆锥": "cone", "扇形": "sector",
        }
        for cn, en in cn_map.items():
            if cn in combined:
                keywords.add(en)

        # 限制数量，优先英文术语
        result = list(keywords)[:max_keywords]
        return result if result else ["geometry", "diagram", "figure"]

    # ---------- 第二阶段：图像 RAG 精筛选 ----------
    def image_rag_refine(
        self,
        candidates: list[dict],
        user_image_path: str,
        targeted_keywords: list[str],
        top_k: int = IMAGE_TOP_K,
    ) -> list[dict]:
        """
        图像 RAG 精筛：结合用户图像相似度 + 靶向词-图像语义匹配
        """
        self._load_models()
        self._load_store()

        candidate_ids = [c["id"] for c in candidates]
        existing = self._store.image_get(candidate_ids, include=["embeddings", "metadatas"])

        if not existing["ids"]:
            return candidates[:top_k]

        id_to_idx = {id_: i for i, id_ in enumerate(existing["ids"])}
        candidate_embeddings = torch.tensor(existing["embeddings"], device=self.device)

        # 用户图像向量
        has_user_image = False
        if user_image_path:
            if not os.path.exists(user_image_path):
                user_image_path = get_image_full_path(user_image_path)
            has_user_image = os.path.exists(user_image_path)

        if has_user_image:
            img = Image.open(user_image_path).convert("RGB")
            with torch.no_grad():
                img_t = self._clip_preprocess(img).unsqueeze(0).to(self.device)
                user_emb = self._clip_model.encode_image(img_t)
                user_emb = user_emb / user_emb.norm(dim=-1, keepdim=True)
            img_sim = (user_emb @ candidate_embeddings.T).squeeze(0)
        else:
            img_sim = torch.zeros(candidate_embeddings.shape[0], device=self.device)

        # 靶向词文本向量（CLIP 文本编码）
        with torch.no_grad():
            kw_text = " ".join(targeted_keywords) if targeted_keywords else "geometry diagram"
            kw_tokens = open_clip.tokenize([kw_text]).to(self.device)
            kw_emb = self._clip_model.encode_text(kw_tokens)
            kw_emb = kw_emb / kw_emb.norm(dim=-1, keepdim=True)
        kw_sim = (kw_emb @ candidate_embeddings.T).squeeze(0)

        # 融合：有用户图时图像为主+靶向词为辅；无用户图时仅靶向词
        if has_user_image:
            fused = IMAGE_SIM_WEIGHT * img_sim + KEYWORD_SIM_WEIGHT * kw_sim
        else:
            fused = kw_sim

        # 排序取 Top-K
        scores, indices = torch.sort(fused, descending=True)
        top_indices = indices[:top_k].cpu().tolist()

        refined = []
        for j, idx in enumerate(top_indices):
            rid = existing["ids"][idx]
            for c in candidates:
                if c["id"] == rid:
                    c_copy = dict(c)
                    c_copy["image_rag_score"] = float(scores[j])
                    refined.append(c_copy)
                    break
        return refined

    # ---------- 最终聚合回答 ----------
    def aggregate_answer(
        self,
        user_question: str,
        top_results: list[dict],
        targeted_keywords: list[str],
        user_image_path: str = "",
    ) -> str:
        """
        基于 Top-3 结果聚合生成回答。
        若配置了 LLM，则生成贴合用户场景的自然语言回答；否则返回检索结果模板。
        """
        if not top_results:
            return "未找到相关题目，请尝试换一种表述或上传更清晰的题目图像。"

        # 优先使用 LLM 生成贴合用户场景的回答（传入用户图片供视觉模型识别）
        llm_answer = generate_answer(user_question, top_results, targeted_keywords, user_image_path)
        if llm_answer:
            ref_lines = ["\n\n---\n**参考题目**（检索到的相似题）:"]
            for i, r in enumerate(top_results[:3], 1):
                md = r.get("metadata", {})
                q = md.get("question", "")[:80]
                ans = md.get("answer", "")
                ref_lines.append(f"  {i}. {q}... 答案: {ans}")
            return llm_answer + "\n" + "\n".join(ref_lines)

        # 无 LLM 时返回模板
        parts = ["## 检索到的相关题目与知识点\n"]
        for i, r in enumerate(top_results, 1):
            md = r.get("metadata", {})
            q = md.get("question", "")
            kc = md.get("knowledge_concept", "")
            ans = md.get("answer", "")
            opt = md.get("option", "")
            parts.append(f"### 题目 {i}\n")
            parts.append(f"**题目**: {q[:200]}...\n" if len(q) > 200 else f"**题目**: {q}\n")
            parts.append(f"**知识点**: {kc}\n")
            parts.append(f"**选项**: {opt[:150]}...\n" if len(opt) > 150 else f"**选项**: {opt}\n")
            parts.append(f"**答案**: {ans}\n\n")
        parts.append("\n## 靶向关键词\n")
        parts.append(", ".join(targeted_keywords))
        parts.append("\n\n基于以上检索结果，建议结合题目图像与知识点进行解答。")
        return "".join(parts)

    # ---------- 主流程 ----------
    def run(
        self,
        user_question: str,
        user_image_path: str | None = None,
        text_top_k: int = TEXT_TOP_K,
        image_top_k: int = IMAGE_TOP_K,
    ) -> dict:
        """
        完整两阶段 RAG 流程
        返回: {
            "candidates_text": 文本粗筛结果,
            "targeted_keywords": 靶向词,
            "top_results": 图像精筛 Top-K,
            "answer": 聚合回答
        }
        """
        # 1. 文本 RAG 粗筛
        candidates = self.text_rag_retrieve(user_question, top_k=text_top_k)

        # 2. 靶向词生成
        targeted_keywords = self.generate_targeted_keywords(candidates)

        # 3. 图像 RAG 精筛（无用户图时仍可用靶向词重排）
        user_image_path = user_image_path or ""
        top_results = self.image_rag_refine(
            candidates, user_image_path, targeted_keywords, top_k=image_top_k
        )

        # 4. 聚合回答（传入用户图片，供 LLM 视觉模型识别）
        answer = self.aggregate_answer(user_question, top_results, targeted_keywords, user_image_path)

        return {
            "candidates_text": candidates,
            "targeted_keywords": targeted_keywords,
            "top_results": top_results,
            "answer": answer,
        }


if __name__ == "__main__":
    agent = RAGAgent()
    # 示例查询
    result = agent.run(
        user_question="正方形顶点为圆心画圆，求圆心角的度数",
        user_image_path="",  # 可选：用户上传的题目图
    )
    print(result["answer"])
    print("\n靶向词:", result["targeted_keywords"])
    print("\nTop-3 结果数:", len(result["top_results"]))
