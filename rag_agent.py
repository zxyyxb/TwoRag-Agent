"""
两阶段多模态 RAG Agent
由 LLM 智能体自主选择：直接回复（如问候）或调用工具链（文本 RAG → 靶向词 → 图像 RAG → 聚合回答）。
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
    REACT_MODE,
    REACT_LOG_DIR,
)
from vector_store import NumpyVectorStore
from answer_generator import generate_answer, react_agent_decide, generate_direct_reply
from react_logger import ReactLogger


def get_image_full_path(image_path: str) -> str:
    if not image_path:
        return ""
    if os.path.isabs(image_path):
        return image_path
    base = os.path.basename(image_path)
    return os.path.join(IMAGE_BASE_DIR, base)


def _need_full_rag(user_question: str, has_image: bool) -> bool:
    """用户上传了图片或问题表述像在问题目时，需要走完整双 RAG，不能直接聚合。"""
    if has_image:
        return True
    q = (user_question or "").strip()
    if not q:
        return False
    if len(q) > 15:
        return True
    question_marks = ("题", "求", "解", "怎么", "多少", "什么", "如何", "？", "?")
    return any(m in q for m in question_marks)


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
    def generate_targeted_keywords(self, candidates: list[dict], user_question: str, max_keywords: int = 10) -> list[str]:
        """
        基于「用户问题 + 文本 RAG 候选数据」共同提炼靶向词（关键词/主题/概念），
        用户问题在整体语义中权重更高，用于辅助图像 RAG 做更精准的语义对齐。
        """
        # 1. 聚合候选集中的题目、知识点、答案
        all_texts = []
        for c in candidates[:min(15, len(candidates))]:
            md = c.get("metadata", {})
            doc = c.get("document", "")
            all_texts.append(doc or "")
            all_texts.append(md.get("knowledge_concept", ""))
            all_texts.append(md.get("question", ""))

        # 2. 将用户问题与候选文本拼接；通过重复用户问题提高其权重
        uq = (user_question or "").strip()
        combined_parts = []
        if uq:
            # 用户表述重复多次，相当于在规则匹配里提高其占比
            combined_parts.extend([uq] * 3)
        combined_parts.extend(filter(None, all_texts))
        combined = " ".join(combined_parts)

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

        # 3. 中文关键概念映射（常见几何），同样考虑用户问题中的中文描述
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
        基于 Top 结果聚合生成回答。无论是否检索到相关题目都会调用大模型生成答案，
        通过提示词约束模型不编造条件。
        """
        # 始终调用 LLM 生成回答（无结果或匹配度低时由提示词约束模型不编造）
        llm_answer = generate_answer(user_question, top_results, targeted_keywords, user_image_path)
        if llm_answer:
            if top_results:
                ref_lines = ["\n\n---\n**参考题目**（检索到的相似题）:"]
                for i, r in enumerate(top_results[:3], 1):
                    md = r.get("metadata", {})
                    q = md.get("question", "")[:80]
                    ans = md.get("answer", "")
                    ref_lines.append(f"  {i}. {q}... 答案: {ans}")
                return llm_answer + "\n" + "\n".join(ref_lines)
            return llm_answer

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

    # ---------- 额外工具：为用户推荐相关练习题 ----------
    def recommend_exercises(
        self,
        user_question: str,
        user_image_path: str | None = None,
        n_exercises: int = 5,
        text_top_k: int = TEXT_TOP_K,
        image_top_k: int = IMAGE_TOP_K,
    ) -> list[dict]:
        """
        根据「用户当前的问题/薄弱知识点 + 题库」推荐若干道相似练习题。

        设计思路：
        - 仍然走“两阶段 RAG”链路：文本 RAG → 靶向词生成 → 图像 RAG 精筛；
        - 但只返回题目本身的信息（题干 / 知识点 / 选项 / 答案 / 图片路径等），不调用 LLM 生成讲解；
        - 供上层对话 Agent 在用户说「再来几道类似的题」「这块再出几道练习」时作为工具调用。

        多轮对话建议：
        - 上层对话管理逻辑可以记录最近一次 user_question；
        - 当用户说「再来几道类似的题」时，直接用上一次的问题再次调用本方法即可；
        - 也可以在用户只说「圆柱体积再出几道题」时，把这句话作为 user_question，单独调用本方法。
        """
        user_image_path = user_image_path or ""

        # 文本 RAG 粗召回
        candidates = self.text_rag_retrieve(user_question, top_k=text_top_k)
        # 基于「用户问题 + 候选文本」生成靶向词（用户问题占比更大）
        targeted_keywords = self.generate_targeted_keywords(candidates, user_question)
        # 图像 RAG 精筛（如果有用户图像则会参与排序）
        refined = self.image_rag_refine(
            candidates,
            user_image_path=user_image_path,
            targeted_keywords=targeted_keywords,
            top_k=image_top_k,
        )

        # 组装给上层使用的练习题结构，只保留核心信息，最多 n_exercises 道
        exercises: list[dict] = []
        for item in refined[: max(0, n_exercises)]:
            md = item.get("metadata", {}) or {}
            exercises.append(
                {
                    "id": item.get("id"),
                    "question": md.get("question", ""),
                    "knowledge_concept": md.get("knowledge_concept", ""),
                    "option": md.get("option", ""),
                    "answer": md.get("answer", ""),
                    "image_path": md.get("image_path", ""),
                    # 兼容可能存在的行号/原始 ID 字段，方便前端回溯原数据
                    "source_row": md.get("row_id", md.get("id", "")),
                    # 若 image_rag_refine 写入了图像打分，可作为相似度参考
                    "image_rag_score": item.get("image_rag_score"),
                }
            )
        return exercises

    # ---------- ReAct 工具：每个工具接收 state，返回 (state 更新, thought, action, observation) ----------
    def _tool_text_rag(self, state: dict, text_top_k: int, image_top_k: int) -> tuple[dict, str, str, str]:
        """工具：文本 RAG 粗检索"""
        candidates = self.text_rag_retrieve(state["user_question"], top_k=text_top_k)
        cand_summary = ", ".join([c.get("id", "") for c in candidates[:5]]) + (f" 等共 {len(candidates)} 条" if len(candidates) > 5 else "")
        cand_topics = ", ".join([str(c.get("metadata", {}).get("knowledge_concept", ""))[:30] for c in candidates[:3]])
        # 为日志提供更清晰的题目级信息，便于事后复盘
        top_questions = []
        for i, c in enumerate(candidates[:3], start=1):
            md = c.get("metadata", {})
            q = (md.get("question") or "").strip()
            if q:
                top_questions.append(f"{i}. {q[:160]}{'...' if len(q) > 160 else ''}")
        thought = "需要先在文本向量库中检索与用户问题语义相关的候选题目，缩小搜索范围。"
        action = "text_rag_retrieve(question, top_k={})".format(text_top_k)
        observation = "召回 {} 条候选。ID: {}。涉及知识点: {}。Top-3 题目预览: {}".format(
            len(candidates),
            cand_summary,
            cand_topics,
            " | ".join(top_questions) if top_questions else "(无可展示题目文本)",
        )
        return {"candidates": candidates}, thought, action, observation

    def _tool_generate_keywords(self, state: dict) -> tuple[dict, str, str, str]:
        """工具：从候选生成靶向词"""
        candidates = state.get("candidates", [])
        targeted_keywords = self.generate_targeted_keywords(candidates, state.get("user_question", ""))
        thought = "综合用户问题与文本召回的真实知识数据提炼靶向词，其中用户问题占比更大，以兼顾用户意图与题库语义，避免检索漂移。"
        action = "generate_targeted_keywords(candidates, user_question)"
        observation = "生成靶向词: {}".format(targeted_keywords)
        return {"targeted_keywords": targeted_keywords}, thought, action, observation

    def _tool_image_rag(self, state: dict, image_top_k: int) -> tuple[dict, str, str, str]:
        """工具：图像 RAG 精筛"""
        candidates = state.get("candidates", [])
        top_results = self.image_rag_refine(
            candidates,
            state.get("user_image_path", ""),
            state.get("targeted_keywords", []),
            top_k=image_top_k,
        )
        top_summary = "; ".join([
            "({}... 答案:{})".format(str(r.get("metadata", {}).get("question", ""))[:40], r.get("metadata", {}).get("answer", ""))
            for r in top_results
        ])
        thought = "用靶向词 + 用户图像对候选集做图像向量相似度计算，融合后重排取 Top-K。"
        action = "image_rag_refine(candidates, user_image_path, targeted_keywords, top_k={})".format(image_top_k)
        observation = "精筛得到 Top-{}: {}".format(len(top_results), top_summary)
        return {"top_results": top_results}, thought, action, observation

    def _tool_aggregate_answer(self, state: dict) -> tuple[dict, str, str, str]:
        """工具：聚合生成最终回答"""
        answer = self.aggregate_answer(
            state["user_question"],
            state.get("top_results", []),
            state.get("targeted_keywords", []),
            state.get("user_image_path", ""),
        )
        thought = "将 Top 检索结果作为上下文，结合用户问题与靶向词，组织生成可解释的最终回答（或条件不足时直接返回无解）。"
        action = "aggregate_answer(question, top_results, targeted_keywords, user_image_path)"
        # 在日志中记录部分答案内容，便于快速浏览 ReAct 过程中“针对题目给出的解答”
        preview_len = 300
        answer_preview = (answer or "")[:preview_len]
        observation = "已生成回答，长度 {} 字符。回答前 {} 字符预览: {}".format(len(answer), preview_len, answer_preview)
        return {"answer": answer}, thought, action, observation

    def _get_react_tools(self):
        """返回 动作名 -> (步骤显示名, 工具函数)，供智能体按 LLM 选择的 action 调用。"""
        return {
            "text_rag_retrieve": ("1. 文本 RAG 粗召回", lambda s, tk, ik: self._tool_text_rag(s, tk, ik)),
            "generate_targeted_keywords": ("2. 靶向词生成", lambda s, tk, ik: self._tool_generate_keywords(s)),
            "image_rag_refine": ("3. 图像 RAG 精筛", lambda s, tk, ik: self._tool_image_rag(s, ik)),
            "aggregate_answer": ("4. 聚合生成回答", lambda s, tk, ik: self._tool_aggregate_answer(s)),
        }

    # ---------- 主流程：由 LLM 智能体选择工具或直接回复（ReAct） ----------
    def run(
        self,
        user_question: str,
        user_image_path: str | None = None,
        text_top_k: int = TEXT_TOP_K,
        image_top_k: int = IMAGE_TOP_K,
        react_mode: bool | None = None,
    ) -> dict:
        """
        由 LLM 智能体决定每一步：选择「直接回复」或调用某一工具（文本 RAG、靶向词、图像 RAG、聚合回答）。
        例如用户只说「你好」时，智能体会选择 direct_reply 并结束；有具体题目时会选择 RAG 工具链。
        返回: {
            "candidates_text": 文本粗筛结果,
            "targeted_keywords": 靶向词,
            "top_results": 图像精筛 Top-K,
            "answer": 最终回答,
            "react_log_path": ReAct 日志文件路径（若启用）
        }
        """
        user_image_path = user_image_path or ""
        use_react = react_mode if react_mode is not None else REACT_MODE
        logger = ReactLogger(log_dir=REACT_LOG_DIR, enabled=use_react)
        logger.start(user_question, user_image_path)

        state = {
            "user_question": user_question or "",
            "user_image_path": user_image_path,
            "candidates": [],
            "targeted_keywords": [],
            "top_results": [],
            "answer": "",
        }
        tools = self._get_react_tools()
        history: list[tuple[str, str, str]] = []
        max_steps = 10

        for _ in range(max_steps):
            has_candidates = len(state.get("candidates", [])) > 0
            has_keywords = len(state.get("targeted_keywords", [])) > 0
            has_top_results = len(state.get("top_results", [])) > 0
            has_answer = bool(state.get("answer", "").strip())
            has_image = bool(state.get("user_image_path", "").strip())
            need_full_rag = _need_full_rag(state.get("user_question", ""), has_image)

            thought, action = react_agent_decide(
                user_question=state["user_question"],
                has_image=has_image,
                has_candidates=has_candidates,
                has_keywords=has_keywords,
                has_top_results=has_top_results,
                has_answer=has_answer,
                history=history,
            )
            # 状态校验：有图或题目相关表述时必须走完双 RAG，不得选 direct_reply 或提前 aggregate_answer
            if need_full_rag and not has_top_results:
                if action == "direct_reply" or action == "aggregate_answer" or (action == "image_rag_refine" and not has_keywords) or (action == "generate_targeted_keywords" and not has_candidates):
                    if not has_candidates:
                        action = "text_rag_retrieve"
                        thought = thought or "用户有图或题目相关表述，需先执行文本 RAG 召回。"
                    elif not has_keywords:
                        action = "generate_targeted_keywords"
                        thought = thought or "已有文本候选，需先生成靶向词。"
                    else:
                        action = "image_rag_refine"
                        thought = thought or "已有靶向词，需执行图像 RAG 精筛。"
            elif not action or action not in ("direct_reply", "text_rag_retrieve", "generate_targeted_keywords", "image_rag_refine", "aggregate_answer"):
                # 解析失败时按当前状态推断下一步，避免误选 direct_reply 导致流程中断、最终答案为空
                if need_full_rag:
                    if not has_candidates:
                        action = "text_rag_retrieve"
                    elif not has_keywords:
                        action = "generate_targeted_keywords"
                    elif not has_top_results:
                        action = "image_rag_refine"
                    else:
                        action = "aggregate_answer"
                else:
                    action = "direct_reply"
                thought = thought or "根据当前状态选择下一步。"

            logger.step("智能体决策", thought, "Action: " + action, "")

            if action == "direct_reply":
                answer = generate_direct_reply(state["user_question"])
                state["answer"] = answer
                logger.step("直接回复", "用户无需检索，由 LLM 生成简短回复。", "direct_reply(user_question)", "已生成回复，长度 {} 字符。".format(len(answer)))
                break

            if action not in tools:
                break
            step_name, tool_fn = tools[action]
            updates, t, a, obs = tool_fn(state, text_top_k, image_top_k)
            state.update(updates)
            history.append((t, a, obs))
            logger.step(step_name, t, a, obs)
            if action == "aggregate_answer":
                break

        logger.final(state["answer"])

        result = {
            "candidates_text": state["candidates"],
            "targeted_keywords": state["targeted_keywords"],
            "top_results": state["top_results"],
            "answer": state["answer"],
        }
        if use_react and logger.log_path:
            result["react_log_path"] = logger.log_path
        return result


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
