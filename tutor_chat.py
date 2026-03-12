"""
交互式对话式数学辅导助手

功能：
- 用户可以多轮和 Agent 对话，提问任意数学题或知识点；
- 用户可以补充自己薄弱的知识点，比如：
    「我圆柱体积这块比较薄弱，帮我系统讲讲并出几道题」；
- 在任意一道题目讲解之后，用户可以让 Agent 推荐几道类似的题目练习。

用法示例（命令行）：
    python tutor_chat.py
"""

from __future__ import annotations

from typing import List, Dict, Any

from rag_agent import RAGAgent


def _print_recommended_questions(top_results: List[Dict[str, Any]], max_n: int = 5) -> None:
    """根据上一轮检索结果，推荐若干道类似题目给用户练习。"""
    if not top_results:
        print("当前没有可用的相似题目结果，请先提一道题目。")
        return

    n = min(max_n, len(top_results))
    print("\n" + "=" * 60)
    print(f"【为你推荐的 {n} 道类似题目】")
    print("=" * 60)

    for i, r in enumerate(top_results[:n], start=1):
        md = r.get("metadata", {}) or {}
        q = (md.get("question") or "").strip()
        kc = (md.get("knowledge_concept") or "").strip()
        opt = (md.get("option") or "").strip()
        ans = (md.get("answer") or "").strip()

        print(f"\n第 {i} 题：")
        if q:
            print("题目：", q)
        if kc:
            print("知识点：", kc)
        if opt:
            # 练习模式下，一般希望先做题再看答案，这里仅展示选项
            print("选项：", opt)
        if ans:
            print("【参考答案】", ans)

    print("\n（提示：你也可以根据这些题目继续提问，例如“帮我详细讲讲第 1 题怎么做”。）")


def interactive_chat() -> None:
    """命令行多轮对话入口。"""
    agent = RAGAgent()
    last_top_results: List[Dict[str, Any]] = []

    print("=" * 60)
    print("两阶段多模态 RAG 数学辅导助手（对话模式）")
    print("=" * 60)
    print("说明：")
    print("- 直接输入你的题目或薄弱知识点，例如：")
    print("  「这道题怎么做？」「我对圆柱体积这块比较薄弱，帮我系统讲讲并出几道题」")
    print("- 在讲解完一题之后，若想多练几道类似的题，可以输入：")
    print("  「推荐几道类似的题」「再来几道相似的练习题」")
    print("- 输入 exit / quit / q / 退出 可以结束对话。")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话。")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in {"exit", "quit", "q"} or user_input in {"退出", "结束"}:
            print("好的，下次再见～")
            break

        # 用户显式请求「推荐类似题目」：优先使用上一轮的检索结果
        if ("类似" in user_input and "题" in user_input) or ("推荐" in user_input and "题" in user_input):
            _print_recommended_questions(last_top_results)
            continue

        # 正常问答 / 知识点讲解
        print("\n助手：正在思考并检索相关题目，请稍候...")
        result = agent.run(
            user_question=user_input,
            user_image_path="",  # 对话模式默认不带图；如需带图可后续扩展命令
        )

        answer = result.get("answer", "") or ""
        print("\n" + "=" * 60)
        print("【助手回答】")
        print("=" * 60)
        print(answer)

        last_top_results = result.get("top_results") or []
        if last_top_results:
            print("\n（如果你想多练习，可以输入：推荐几道类似的题）")


if __name__ == "__main__":
    interactive_chat()

