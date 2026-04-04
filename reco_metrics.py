"""
推荐 / 检索类任务的离线指标：HR@K、Recall@K、NDCG@K、MRR 等。
定义与信息检索 / 推荐系统常见用法一致，便于论文表述。
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _binary_relevance(
    ranked_ids: Sequence[str],
    relevant: set[str],
    k: int,
) -> list[int]:
    """前 k 个位置上的二值相关性 rel_i ∈ {0,1}。"""
    top = list(ranked_ids)[:k]
    return [1 if rid in relevant else 0 for rid in top]


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """
    HR@K（命中率）：若 Top-K 中至少出现 1 个相关项则为 1，否则为 0。
    对多样本取平均即常见论文中的 Hit Rate@K。
    """
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return 1.0 if (top & relevant) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Recall@K = |Rel ∩ TopK| / |Rel|。"""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Precision@K = |Rel ∩ TopK| / K（分母固定为 K，不足 K 个时仍除以 K）。"""
    if k <= 0:
        return 0.0
    top = list(ranked_ids)[:k]
    hits = sum(1 for rid in top if rid in relevant)
    return hits / k


def dcg_at_k(rels: Sequence[float], k: int) -> float:
    """DCG@K，位置从 1 开始：sum rel_i / log2(i+1)。"""
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel == 0:
            continue
        s += rel / math.log2(i + 1)
    return s


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant: set[str],
    k: int,
    graded_relevance: Mapping[str, float] | None = None,
) -> float:
    """
    NDCG@K。
    - 若 graded_relevance 为 None：二值相关（在 relevant 集合内为 1，否则 0）。
    - 若提供 graded_relevance：用其作为每个 id 的相关性等级（如文本相似度分档）。
    """
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0

    if graded_relevance is None:
        rels = [1.0 if rid in relevant else 0.0 for rid in top]
        # 理想情况：所有相关项排在最前；IDCG 用全局 relevant 数量截断到 k
        n_rel = min(len(relevant), k)
        if n_rel == 0:
            return 0.0
        ideal_rels = [1.0] * n_rel + [0.0] * (k - n_rel)
    else:
        rels = [float(graded_relevance.get(rid, 0.0)) for rid in top]
        all_ids = set(relevant) | set(top)
        scores = [graded_relevance.get(rid, 0.0) for rid in all_ids]
        scores.sort(reverse=True)
        ideal_rels = scores[:k]
        if not ideal_rels or max(ideal_rels) <= 0:
            return 0.0

    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """MRR：第一个相关项的位次的倒数；无相关项则为 0。"""
    if not relevant:
        return 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def aggregate_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def mean_score_at_k(
    ranked_ids: Sequence[str],
    id_to_score: Mapping[str, float],
    k: int,
    skip_missing: bool = False,
) -> float:
    """
    Top-K 平均得分（如图像相似度、多模态打分）：用于 MIS@K 等。
    skip_missing=False 时，缺失 id 记为 0；True 时跳过不参与分母。
    """
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0
    vals: list[float] = []
    for rid in top:
        if rid not in id_to_score:
            if skip_missing:
                continue
            vals.append(0.0)
        else:
            vals.append(float(id_to_score[rid]))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def ndcg_at_k_oracle_pool(
    ranked_ids: Sequence[str],
    pool_grades: Mapping[str, float],
    k: int,
) -> float:
    """
    给定**固定候选池**上每个 item 的相关性等级（如多模态加权分），
    对排序结果算 NDCG@K；理想排序 DCG 由池中分数最高的 K 个构成（客观上界）。
    """
    if not pool_grades or k <= 0:
        return 0.0
    ideal_vals = sorted(pool_grades.values(), reverse=True)[:k]
    if not ideal_vals or max(ideal_vals) <= 0:
        return 0.0
    top = [rid for rid in ranked_ids if rid in pool_grades][:k]
    rels = [float(pool_grades.get(rid, 0.0)) for rid in top]
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal_vals, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg
