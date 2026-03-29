"""
离线评估「练习题推荐 / 双阶段 RAG」排序质量。

默认标签（弱监督）：同一 knowledge_concept（与 build_vectors 写入 metadata 的字段一致）视为相关，
排除查询自身。合并题库下 WeMath 为数学知识点、ScienceQA 为 category 字段，均写入 knowledge_concept。

指标：HR@K、Recall@K、Precision@K、NDCG@K（二值相关）、MRR；K 默认 5 与 10。
另提供文本单路排序基线，以及 Top-K 上「文本相似度 vs 图像相似度」的分解统计（对应导师建议的区分维度）。

用法（需已运行 build_vectors.py 生成向量库）:
    python eval_recommendation.py --max-queries 300 --seed 42
    python eval_recommendation.py --no-query-image   # 消融：无用户图
    python eval_recommendation.py --query-mode question
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

from config import CHROMA_PERSIST_DIR, TEXT_TOP_K
from rag_agent import RAGAgent, get_image_full_path
from reco_metrics import (
    aggregate_mean,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from vector_store import NumpyVectorStore


def _normalize_kc(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _build_kc_index(store: NumpyVectorStore) -> tuple[dict[str, str], dict[str, int]]:
    """row_id -> 规范化知识点；row_id -> 文本向量行下标"""
    id_to_kc: dict[str, str] = {}
    id_to_tidx: dict[str, int] = {}
    for i, rid in enumerate(store.text_ids):
        md = store.text_metadatas[i] if i < len(store.text_metadatas) else {}
        id_to_kc[rid] = _normalize_kc(str(md.get("knowledge_concept", "") or ""))
        id_to_tidx[rid] = i
    return id_to_kc, id_to_tidx


def _query_text_from_meta(md: dict, mode: str) -> str:
    q = str(md.get("question", "") or "").strip()
    kc = str(md.get("knowledge_concept", "") or "").strip()
    desc = str(md.get("knowledge_concept_desc", "") or "").strip()[:800]
    if mode == "question":
        return q if q else kc
    # full：贴近索引文档，但评估「只贴题干」时可换 question 模式
    parts = [p for p in (q, kc, desc) if p]
    return "\n".join(parts) if parts else " "


def _ranked_ids_dual_rag(
    agent: RAGAgent,
    user_question: str,
    user_image_path: str,
    text_top_k: int,
    image_top_k: int,
) -> list[str]:
    candidates = agent.text_rag_retrieve(user_question, top_k=text_top_k)
    kw = agent.generate_targeted_keywords(candidates, user_question)
    refined = agent.image_rag_refine(
        candidates,
        user_image_path=user_image_path,
        targeted_keywords=kw,
        top_k=image_top_k,
    )
    return [c["id"] for c in refined]


def _ranked_ids_text_only(
    store: NumpyVectorStore,
    query_emb: np.ndarray,
    exclude_id: str,
    k: int,
) -> list[str]:
    n = min(max(k, 50), store.text_count())
    res = store.text_query(query_emb, n_results=n)
    ids = res["ids"][0] if res["ids"] else []
    out: list[str] = []
    for rid in ids:
        if rid == exclude_id:
            continue
        out.append(rid)
        if len(out) >= k:
            break
    return out


def _text_cosine_matrix(
    store: NumpyVectorStore,
    id_to_tidx: dict[str, int],
    query_id: str,
    cand_ids: list[str],
) -> list[float]:
    """查询项与候选的文本余弦相似度（向量已 L2 归一化时为点积）。"""
    qi = id_to_tidx.get(query_id)
    if qi is None:
        return [0.0] * len(cand_ids)
    qv = store.text_embeddings[qi].astype(np.float32)
    sims: list[float] = []
    for rid in cand_ids:
        j = id_to_tidx.get(rid)
        if j is None:
            sims.append(0.0)
            continue
        v = store.text_embeddings[j]
        sims.append(float(np.dot(qv, v)))
    return sims


def _image_cosine_matrix(
    agent: RAGAgent,
    store: NumpyVectorStore,
    query_id: str,
    cand_ids: list[str],
) -> list[float]:
    """查询项图与候选图的 CLIP 图像向量余弦相似度；缺图则为 nan。"""
    agent._load_models()
    agent._load_store()
    device = agent.device
    clip_model = agent._clip_model
    preprocess = agent._clip_preprocess

    def emb_for_row(rid: str) -> torch.Tensor | None:
        if rid not in store._id_to_img_idx:
            return None
        idx = store._id_to_img_idx[rid]
        t = torch.tensor(store.image_embeddings[idx], device=device, dtype=torch.float32)
        t = t / (t.norm() + 1e-8)
        return t

    qe = emb_for_row(query_id)
    out: list[float] = []
    for rid in cand_ids:
        ce = emb_for_row(rid)
        if qe is None or ce is None:
            out.append(float("nan"))
        else:
            out.append(float((qe * ce).sum().item()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="双 RAG 推荐排序离线评估")
    parser.add_argument("--persist-dir", type=str, default=CHROMA_PERSIST_DIR, help="向量库目录")
    parser.add_argument("--max-queries", type=int, default=500, help="最多评估的查询条数（子采样）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10], help="K 列表，如 5 10")
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=max(TEXT_TOP_K, 20),
        help="第一阶段文本召回上限（需 >= 评估用 K）",
    )
    parser.add_argument(
        "--image-top-k",
        type=int,
        default=20,
        help="图像精筛保留的候选数（用于算到 K=10 的指标）",
    )
    parser.add_argument(
        "--query-mode",
        choices=("question", "full"),
        default="question",
        help="查询文本：仅题干 question / 与建库接近的 full",
    )
    parser.add_argument(
        "--no-query-image",
        action="store_true",
        help="不传用户图（测无图场景）；默认对有图样本使用题库中该题配图模拟上传",
    )
    parser.add_argument(
        "--min-relevant-peers",
        type=int,
        default=1,
        help="同一知识点下除自身外至少要有多少题才纳入评估",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="将汇总指标写入 JSON 文件",
    )
    args = parser.parse_args()
    ks = sorted(set(args.k))

    if not os.path.isdir(args.persist_dir):
        print(f"向量库不存在: {args.persist_dir}", file=sys.stderr)
        sys.exit(1)

    store = NumpyVectorStore(args.persist_dir)
    store.load()
    id_to_kc, id_to_tidx = _build_kc_index(store)

    kc_to_ids: dict[str, list[str]] = defaultdict(list)
    for rid, kc in id_to_kc.items():
        if kc:
            kc_to_ids[kc].append(rid)

    eligible: list[str] = []
    for rid, kc in id_to_kc.items():
        if not kc:
            continue
        peers = [x for x in kc_to_ids[kc] if x != rid]
        if len(peers) < args.min_relevant_peers:
            continue
        eligible.append(rid)

    if not eligible:
        print("没有满足条件的查询（请检查 metadata 中 knowledge_concept 是否非空）", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(eligible)
    query_ids = eligible[: min(args.max_queries, len(eligible))]

    print(f"向量库: {args.persist_dir}")
    print(f"可评估查询数（有同知识点邻居）: {len(eligible)}，本次使用: {len(query_ids)}")
    print(f"K = {ks}，query_mode = {args.query_mode}，用户图 = {not args.no_query_image}")

    agent = RAGAgent()

    # 逐指标累加
    metrics_dual = {f"hr@{k}": [] for k in ks}
    metrics_dual.update({f"recall@{k}": [] for k in ks})
    metrics_dual.update({f"precision@{k}": [] for k in ks})
    metrics_dual.update({f"ndcg@{k}": [] for k in ks})
    metrics_mrr_dual: list[float] = []

    metrics_txt = {f"hr@{k}": [] for k in ks}
    metrics_txt.update({f"recall@{k}": [] for k in ks})
    metrics_txt.update({f"precision@{k}": [] for k in ks})
    metrics_txt.update({f"ndcg@{k}": [] for k in ks})
    metrics_mrr_txt: list[float] = []

    # 分解：双 RAG Top-10 上 text_sim / image_sim
    sum_text_when_hit: list[float] = []
    sum_img_when_hit: list[float] = []
    sum_text_when_miss: list[float] = []
    sum_img_when_miss: list[float] = []
    n_img_pairs_hit = 0
    n_img_pairs_miss = 0

    k_eval = max(ks)

    for qid in query_ids:
        tidx = id_to_tidx[qid]
        md = store.text_metadatas[tidx]
        qtext = _query_text_from_meta(md, args.query_mode)
        kc = id_to_kc[qid]
        relevant = set(x for x in kc_to_ids[kc] if x != qid)

        img_rel = str(md.get("image_path", "") or "")
        user_img = ""
        if not args.no_query_image and img_rel:
            p = get_image_full_path(img_rel)
            if os.path.isfile(p):
                user_img = p

        ranked_dual = _ranked_ids_dual_rag(
            agent,
            qtext,
            user_img,
            text_top_k=args.text_top_k,
            image_top_k=args.image_top_k,
        )
        ranked_dual = [rid for rid in ranked_dual if rid != qid]

        for k in ks:
            metrics_dual[f"hr@{k}"].append(hit_rate_at_k(ranked_dual, relevant, k))
            metrics_dual[f"recall@{k}"].append(recall_at_k(ranked_dual, relevant, k))
            metrics_dual[f"precision@{k}"].append(precision_at_k(ranked_dual, relevant, k))
            metrics_dual[f"ndcg@{k}"].append(ndcg_at_k(ranked_dual, relevant, k))
        metrics_mrr_dual.append(mrr(ranked_dual, relevant))

        agent._load_models()
        q_emb = agent._text_model.encode([qtext], normalize_embeddings=True)
        ranked_txt = _ranked_ids_text_only(store, q_emb, qid, k=max(50, k_eval + 10))
        for k in ks:
            metrics_txt[f"hr@{k}"].append(hit_rate_at_k(ranked_txt, relevant, k))
            metrics_txt[f"recall@{k}"].append(recall_at_k(ranked_txt, relevant, k))
            metrics_txt[f"precision@{k}"].append(precision_at_k(ranked_txt, relevant, k))
            metrics_txt[f"ndcg@{k}"].append(ndcg_at_k(ranked_txt, relevant, k))
        metrics_mrr_txt.append(mrr(ranked_txt, relevant))

        # 分解统计（取双 RAG 前 min(10, len)）
        topn = ranked_dual[:10]
        if topn:
            tsims = _text_cosine_matrix(store, id_to_tidx, qid, topn)
            isims = _image_cosine_matrix(agent, store, qid, topn)
            for rid, ts, ims in zip(topn, tsims, isims):
                hit = rid in relevant
                if hit:
                    sum_text_when_hit.append(ts)
                    if not np.isnan(ims):
                        sum_img_when_hit.append(ims)
                        n_img_pairs_hit += 1
                else:
                    sum_text_when_miss.append(ts)
                    if not np.isnan(ims):
                        sum_img_when_miss.append(ims)
                        n_img_pairs_miss += 1

    def summarize(label: str, bucket: dict, mrr_list: list[float]) -> dict:
        row: dict = {"name": label}
        for k in ks:
            row[f"HR@{k}"] = round(aggregate_mean(bucket[f"hr@{k}"]), 4)
            row[f"Recall@{k}"] = round(aggregate_mean(bucket[f"recall@{k}"]), 4)
            row[f"Precision@{k}"] = round(aggregate_mean(bucket[f"precision@{k}"]), 4)
            row[f"NDCG@{k}"] = round(aggregate_mean(bucket[f"ndcg@{k}"]), 4)
        row["MRR"] = round(aggregate_mean(mrr_list), 4)
        return row

    sum_dual = summarize("dual_rag_image_refine", metrics_dual, metrics_mrr_dual)
    sum_txt = summarize("text_only_baseline", metrics_txt, metrics_mrr_txt)

    def avg(xs: list[float]) -> float | None:
        return float(sum(xs) / len(xs)) if xs else None

    decomposition = {
        "avg_text_cosine_top10_when_relevant": avg(sum_text_when_hit),
        "avg_text_cosine_top10_when_irrelevant": avg(sum_text_when_miss),
        "avg_image_cosine_top10_when_relevant": avg(sum_img_when_hit) if n_img_pairs_hit else None,
        "avg_image_cosine_top10_when_irrelevant": avg(sum_img_when_miss) if n_img_pairs_miss else None,
        "n_pairs_text": {"relevant": len(sum_text_when_hit), "irrelevant": len(sum_text_when_miss)},
        "n_pairs_image": {"relevant": n_img_pairs_hit, "irrelevant": n_img_pairs_miss},
        "note": "按同一 knowledge_concept 为相关；图像相似度仅在查询与候选均有图时计入。",
    }

    print("\n=== 主方法：双 RAG（文本召回 + 靶向词 + 图像精排）===")
    print(json.dumps(sum_dual, ensure_ascii=False, indent=2))
    print("\n=== 基线：仅文本向量排序 ===")
    print(json.dumps(sum_txt, ensure_ascii=False, indent=2))
    print("\n=== Top-10 分解：逻辑(文本)相似度 vs 图像相似度（按是否同知识点）===")
    print(json.dumps(decomposition, ensure_ascii=False, indent=2))

    if args.output_json:
        out = {
            "config": vars(args),
            "n_queries": len(query_ids),
            "dual_rag": sum_dual,
            "text_baseline": sum_txt,
            "decomposition": decomposition,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n已写入: {args.output_json}")


if __name__ == "__main__":
    main()
