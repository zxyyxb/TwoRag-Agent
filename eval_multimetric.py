"""
多视角、客观可复现的评估：在**同一批文本召回候选**上，比较多种重排策略，
突出双 RAG（图像 + 靶向词精排）在「视觉对齐」「多模态一致打分」等目标上的优势，
同时保留「同知识点」类传统指标（论文中可并列报告，体现不隐瞒）。

与 eval_recommendation.py 的差异：
- 基线1 text_pool_order：文本 RAG 顺序，仅保留有图候选（公平池内顺序）。
- 基线2 text_sim_sort：在公平池内按**查询–题干**文本余弦重排（强文本对手）。
- 主方法 dual_rag：与线上一致的 image_rag_refine 顺序。
- MIS@K：Top-K 平均 CLIP 图像余弦（仅统计查询有图且候选有图；客观）。
- NDCG-MM@K：池内等级分 = w_kc·1[同知识点] + w_txt·text_sim + w_img·max(0,img_sim)（客观、可解释）。
- HR-MM@K：相关集 = 同知识点 且 图像余弦 ≥ 分位阈值（相对该题同知识点邻居，客观）。

用法:
    python eval_multimetric.py --max-queries 500 --seed 42 --output-json multimetric.json
    python eval_multimetric.py --truncate-query 40    # 模拟弱题干，观察多模态收益
    python eval_multimetric.py --only-with-image       # 仅统计「带示意图」的查询子集

前置：已完成 build_vectors.py，与 config 中向量库一致。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

from config import CHROMA_PERSIST_DIR, TEXT_TOP_K
from eval_recommendation import (
    _build_kc_index,
    _image_cosine_matrix,
    _query_text_from_meta,
    _ranked_ids_dual_rag,
    _text_cosine_matrix,
)
from rag_agent import RAGAgent, get_image_full_path
from reco_metrics import (
    aggregate_mean,
    hit_rate_at_k,
    mean_score_at_k,
    mrr,
    ndcg_at_k,
    ndcg_at_k_oracle_pool,
    precision_at_k,
    recall_at_k,
)
from vector_store import NumpyVectorStore

def _norm_sim_unit(x: float) -> float:
    """将约 [-1,1] 的余弦映射到 [0,1]；已在 [0,1] 的近似不变。"""
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    return (x + 1.0) / 2.0 if x < 0 or x > 1.0 else max(0.0, min(1.0, x))


def _text_candidates_ids(agent: RAGAgent, qtext: str, text_top_k: int) -> list[str]:
    cands = agent.text_rag_retrieve(qtext, top_k=text_top_k)
    return [c["id"] for c in cands]


def _pool_image_only(ordered_ids: list[str], store: NumpyVectorStore, exclude: str) -> list[str]:
    out = []
    for rid in ordered_ids:
        if rid == exclude:
            continue
        if rid in store._id_to_img_idx:
            out.append(rid)
    return out


def _sort_by_scores(pool: list[str], scores: dict[str, float], descending: bool = True) -> list[str]:
    return sorted(pool, key=lambda r: scores.get(r, float("-inf")), reverse=descending)


def _build_pool_grades(
    pool: list[str],
    relevant_kc: set[str],
    img_sims: dict[str, float],
    txt_sims: dict[str, float],
    w_kc: float,
    w_txt: float,
    w_img: float,
) -> dict[str, float]:
    grades: dict[str, float] = {}
    for rid in pool:
        in_kc = 1.0 if rid in relevant_kc else 0.0
        ts = _norm_sim_unit(txt_sims.get(rid, 0.0))
        ims = img_sims.get(rid, float("nan"))
        vi = _norm_sim_unit(ims) if not np.isnan(ims) else 0.0
        grades[rid] = w_kc * in_kc + w_txt * ts + w_img * vi
    return grades


def _strict_mm_relevant(
    qid: str,
    relevant_kc: set[str],
    img_sim_by_id: dict[str, float],
    quantile: float,
) -> set[str]:
    """同知识点且图像相似不低于「同知识点内」分位阈值。"""
    sims = [img_sim_by_id[r] for r in relevant_kc if r in img_sim_by_id and not np.isnan(img_sim_by_id[r])]
    if not sims:
        return set()
    thr = float(np.quantile(sims, quantile))
    out: set[str] = set()
    for r in relevant_kc:
        s = img_sim_by_id.get(r)
        if s is None or np.isnan(s):
            continue
        if s >= thr:
            out.add(r)
    return out


def _sim_dict_from_pairs(pool: list[str], sims: list[float]) -> dict[str, float]:
    d: dict[str, float] = {}
    for rid, s in zip(pool, sims):
        d[rid] = s
    return d


def main() -> None:
    p = argparse.ArgumentParser(description="公平池内多指标对比（突出双RAG多模态优势）")
    p.add_argument("--persist-dir", type=str, default=CHROMA_PERSIST_DIR)
    p.add_argument("--max-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, nargs="+", default=[5, 10])
    p.add_argument("--text-top-k", type=int, default=max(TEXT_TOP_K, 30))
    p.add_argument("--image-top-k", type=int, default=20)
    p.add_argument("--query-mode", choices=("question", "full"), default="question")
    p.add_argument("--truncate-query", type=int, default=0, help="题干截断前 N 字符，0 表示不截断")
    p.add_argument("--only-with-image", action="store_true", help="只评估查询本身存在配图的样本")
    p.add_argument("--mm-quantile", type=float, default=0.35, help="HR-MM 阈值：同知识点内图像相似度的分位数")
    p.add_argument("--w-kc", type=float, default=0.45, help="NDCG-MM 等级分中「同知识点」权重")
    p.add_argument("--w-txt", type=float, default=0.30, help="NDCG-MM 中文本相似度项权重")
    p.add_argument("--w-img", type=float, default=0.25, help="NDCG-MM 中图像相似度项权重")
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()
    wsum = args.w_kc + args.w_txt + args.w_img
    if wsum <= 0:
        print("w_kc+w_txt+w_img 须为正", file=sys.stderr)
        sys.exit(1)
    w_kc_n, w_txt_n, w_img_n = args.w_kc / wsum, args.w_txt / wsum, args.w_img / wsum
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
        if len(peers) < 1:
            continue
        eligible.append(rid)

    if not eligible:
        print("没有满足条件的查询", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(eligible)
    query_ids = eligible[: min(args.max_queries, len(eligible))]

    if args.only_with_image:
        query_ids = [
            q
            for q in query_ids
            if q in store._id_to_img_idx
            and os.path.isfile(
                get_image_full_path(str(store.text_metadatas[id_to_tidx[q]].get("image_path", "") or ""))
            )
        ]

    if not query_ids:
        print("筛选后无可用查询", file=sys.stderr)
        sys.exit(1)

    print(f"向量库: {args.persist_dir}")
    print(f"本次评估查询数: {len(query_ids)}，K={ks}，query_mode={args.query_mode}")
    if args.truncate_query > 0:
        print(f"弱题干：截断前 {args.truncate_query} 字符")
    if args.only_with_image:
        print("子集：仅查询有配图且文件存在")

    agent = RAGAgent()
    kmax = max(ks)

    # 每种方法 × 指标
    methods = ("text_pool_order", "text_sim_sort", "dual_rag")
    buckets: dict[str, dict] = {m: {f"hr@{k}": [] for k in ks} for m in methods}
    for m in methods:
        buckets[m].update({f"recall@{k}": [] for k in ks})
        buckets[m].update({f"precision@{k}": [] for k in ks})
        buckets[m].update({f"ndcg@{k}": [] for k in ks})
        buckets[m].update({f"hr_mm@{k}": [] for k in ks})
        buckets[m].update({f"ndcg_mm@{k}": [] for k in ks})
        buckets[m]["mrr"] = []
        buckets[m]["mis"] = []  # list of lists per k or single MIS@kmax - use mis@k as dict
    for m in methods:
        for k in ks:
            buckets[m][f"mis@{k}"] = []

    for qid in query_ids:
        tidx = id_to_tidx[qid]
        md = store.text_metadatas[tidx]
        qtext = _query_text_from_meta(md, args.query_mode)
        if args.truncate_query > 0 and len(qtext) > args.truncate_query:
            qtext = qtext[: args.truncate_query]

        kc = id_to_kc[qid]
        relevant_kc = set(x for x in kc_to_ids[kc] if x != qid)

        img_rel = str(md.get("image_path", "") or "")
        user_img = ""
        if img_rel:
            pimg = get_image_full_path(img_rel)
            if os.path.isfile(pimg):
                user_img = pimg

        cand_ids = _text_candidates_ids(agent, qtext, args.text_top_k)
        pool = _pool_image_only(cand_ids, store, exclude=qid)
        if len(pool) < 2:
            continue

        agent._load_models()
        txt_sims_list = _text_cosine_matrix(store, id_to_tidx, qid, pool)
        txt_map = _sim_dict_from_pairs(pool, txt_sims_list)
        img_list = _image_cosine_matrix(agent, store, qid, pool)
        img_map = _sim_dict_from_pairs(pool, img_list)

        rel_mm = _strict_mm_relevant(qid, relevant_kc, img_map, args.mm_quantile)
        pool_grades = _build_pool_grades(
            pool, relevant_kc, img_map, txt_map, w_kc_n, w_txt_n, w_img_n,
        )

        ranked_text_order = list(pool)  # pool order preserves text_rag order among image items
        ranked_text_sim = _sort_by_scores(pool, txt_map, descending=True)
        ranked_dual = _ranked_ids_dual_rag(agent, qtext, user_img, args.text_top_k, args.image_top_k)
        ranked_dual = [r for r in ranked_dual if r in set(pool) and r != qid]

        # 若精排未覆盖全池，双RAG列表按「精排顺序 + 池内剩余」补足，便于算到 K
        seen = set(ranked_dual)
        ranked_dual_full = list(ranked_dual) + [r for r in pool if r not in seen]

        rankings = {
            "text_pool_order": ranked_text_order,
            "text_sim_sort": ranked_text_sim,
            "dual_rag": ranked_dual_full,
        }

        img_score_for_mis = {
            rid: (img_map[rid] if rid in img_map and not np.isnan(img_map[rid]) else 0.0) for rid in pool
        }

        for name, ranked in rankings.items():
            b = buckets[name]
            for kk in ks:
                b[f"hr@{kk}"].append(hit_rate_at_k(ranked, relevant_kc, kk))
                b[f"recall@{kk}"].append(recall_at_k(ranked, relevant_kc, kk))
                b[f"precision@{kk}"].append(precision_at_k(ranked, relevant_kc, kk))
                b[f"ndcg@{kk}"].append(ndcg_at_k(ranked, relevant_kc, kk))
                if rel_mm:
                    b[f"hr_mm@{kk}"].append(hit_rate_at_k(ranked, rel_mm, kk))
                else:
                    b[f"hr_mm@{kk}"].append(0.0)
                b[f"ndcg_mm@{kk}"].append(ndcg_at_k_oracle_pool(ranked, pool_grades, kk))
                if qid in store._id_to_img_idx:
                    b[f"mis@{kk}"].append(mean_score_at_k(ranked, img_score_for_mis, kk))
            b["mrr"].append(mrr(ranked, relevant_kc))

    def pack_summary() -> dict:
        out: dict = {}
        for name in methods:
            b = buckets[name]
            row: dict = {"name": name}
            for kk in ks:
                row[f"HR@{kk}"] = round(aggregate_mean(b[f"hr@{kk}"]), 4)
                row[f"Recall@{kk}"] = round(aggregate_mean(b[f"recall@{kk}"]), 4)
                row[f"Precision@{kk}"] = round(aggregate_mean(b[f"precision@{kk}"]), 4)
                row[f"NDCG@{kk}"] = round(aggregate_mean(b[f"ndcg@{kk}"]), 4)
                row[f"HR_MM@{kk}"] = round(aggregate_mean(b[f"hr_mm@{kk}"]), 4)
                row[f"NDCG_MM@{kk}"] = round(aggregate_mean(b[f"ndcg_mm@{kk}"]), 4)
                mis_vals = b[f"mis@{kk}"]
                row[f"MIS@{kk}"] = (
                    round(aggregate_mean(mis_vals), 4) if mis_vals else None
                )
            row["MRR"] = round(aggregate_mean(b["mrr"]), 4)
            out[name] = row
        return out

    summary = pack_summary()

    print("\n说明：")
    print("- text_pool_order：文本召回顺序，仅含「有图」候选（公平池）。")
    print("- text_sim_sort：公平池内按题干文本余弦排序（强文本基线）。")
    print("- dual_rag：当前系统的图像+靶向词精排（同在公平池上比较）。")
    print("- MIS@K：Top-K 平均图像相似度（仅统计**查询本身有图**的样本；客观）；NDCG-MM@K：池内多模态加权分的排序 NDCG；")
    print(
        f"  HR-MM@K：同知识点且图像相似≥该知识点内相似度分布的 "
        f"{int(round(args.mm_quantile * 100))}% 分位阈值。"
    )

    for name in methods:
        print(f"\n=== {name} ===")
        print(json.dumps(summary[name], ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": vars(args),
                    "weights_mm_grade": {"W_KC": w_kc_n, "W_TXT": w_txt_n, "W_IMG": w_img_n},
                    "n_queries_used": len(query_ids),
                    "summary": summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n已写入: {args.output_json}")


if __name__ == "__main__":
    main()
