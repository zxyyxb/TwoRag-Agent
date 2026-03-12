"""
对话式数学辅导 Demo（命令行入口 + 工具调度）

能力：
- 使用 `RAGAgent.run(...)` 解题 / 讲解（双 RAG + LLM）；
- 使用 `RAGAgent.recommend_exercises(...)` 基于某个问题/知识点推荐相似练习题；
- 由本文件中的“对话管理层”根据用户输入自动选择调用哪个工具，而不是在业务层写死固定 if-else。

运行（不带图）：
    python dialog_chat.py

运行（带题目图片，解题和推荐练习都会用到该图）：
    python dialog_chat.py --image test.png
    python dialog_chat.py -i path/to/题目图.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Literal

from rag_agent import RAGAgent


DialogAction = Literal["solve", "recommend_from_last", "recommend_from_text", "solve_recommended"]


@dataclass
class DialogState:
    """对话过程中的简单记忆，用于辅助判断调用哪个工具。"""

    last_question: str | None = None          # 最近一次解题/讲解的问题
    last_exercise_seed: str | None = None     # 最近一次用于生成习题推荐的语句
    last_exercises: List[Dict[str, Any]] | None = None  # 最近一次推荐出来的练习题列表
    user_image_path: str = ""                 # 本场对话使用的题目图片路径（可为空）


def decide_action(user_input: str, state: DialogState) -> DialogAction:
    """
    根据当前用户输入 + 对话上下文，推断本轮更合理的动作。

    设计思路（尽量“让 agent 自己判断诉求”）：
    - 默认优先“解题/讲解”（solve），保证用户随时问随时能得到解答；
    - 当用户在表达“想多做题/要练习”时，切换到习题推荐；
    - 利用上下文：若用户说“再来几道类似的”，没有给新问题，则基于上一次的问题推荐。
    """
    text = user_input.strip()
    lower = text.lower()

    # 0. 若用户指明“第几题”，优先认为是在点名某道推荐题，让助手详细讲解那一道题
    #    例如：“第2题怎么做”“详细讲讲第 3 题”“第1题我不会”
    #    这里不做具体序号解析，只返回 solve_recommended，由上层根据 state.last_exercises 解析。
    if ("第" in text and "题" in text) and any(kw in text for kw in ["怎么做", "不会", "讲", "解析", "解答"]):
        return "solve_recommended"

    # 1. 明确表达“想练习 / 要推荐题 / 多做题”的场景 → 倾向推荐练习（只出题列表，不调用 LLM 再讲一遍）
    ask_practice = (
        any(
            kw in text
            for kw in [
                "类似的题", "类似题", "练习题", "多做几道题", "再出几道题", "再出几题", "多练习", "练习一下",
                "相关题目", "相关的题目", "推荐几道", "推荐几题", "来几道题", "给几道题",
            ]
        )
        or (("推荐" in text or "来几道" in text) and "题" in text)  # 如：给我推荐几道相关的题目
        or any(kw in lower for kw in ["practice", "more problems", "more exercises"])
    )
    if ask_practice:
        if state.last_question:
            return "recommend_from_last"
        return "recommend_from_text"

    # 2. 有明显“不会 / 为什么 / 怎么做 / 求 / 算”等求解意图 → 解题/讲解
    ask_solve = any(
        kw in text
        for kw in ["不会", "怎么做", "怎么算", "为什么", "求", "解", "讲讲", "讲一下", "讲一讲"]
    ) or "?" in text or "？" in text
    if ask_solve:
        return "solve"

    # 3. 用户提到“薄弱、掌握不好”等，一般既包含讲解也包含练习的期望：
    #    优先先讲一遍（solve），再由用户决定是否要练习
    weak_kw = any(kw in text for kw in ["薄弱", "掌握不好", "不熟", "不会做这类题"])
    if weak_kw:
        return "solve"

    # 4. 其他情况：默认当作“问题/知识点咨询”，走 solve 更安全
    return "solve"


def print_exercises(exercises: List[Dict[str, Any]]) -> None:
    """以人类友好的格式打印推荐习题列表。"""
    if not exercises:
        print("当前暂时没有检索到合适的相似练习题，可以先给我一道具体的题目或描述你的薄弱点。")
        return

    print("\n" + "=" * 60)
    print(f"【为你推荐的 {len(exercises)} 道相关练习题】")
    print("=" * 60)

    for i, ex in enumerate(exercises, start=1):
        q = (ex.get("question") or "").strip()
        kc = (ex.get("knowledge_concept") or "").strip()
        opt = (ex.get("option") or "").strip()
        ans = (ex.get("answer") or "").strip()

        print(f"\n第 {i} 题：")
        if q:
            print("题目：", q)
        if kc:
            print("知识点：", kc)
        if opt:
            print("选项：", opt)
        if ans:
            print("【参考答案】", ans)

    print("\n（提示：你可以继续指定其中某一题，让我“详细讲讲第 1 题怎么做”。）")


def interactive_chat(user_image_path: str = "") -> None:
    """
    命令行多轮对话入口：
    - 自动根据用户输入决定调用 run / recommend_exercises；
    - user_image_path：本场对话使用的题目图片，解题和推荐练习时都会传给 Agent。
    """
    agent = RAGAgent()
    state = DialogState(user_image_path=user_image_path or "")

    print("=" * 60)
    print("两阶段多模态 RAG 数学辅导助手（对话模式 Demo）")
    print("=" * 60)
    if state.user_image_path:
        print("当前题目图片：", state.user_image_path)
    print("说明：")
    print("- 可以直接输入任意题目或数学疑问，例如：「这道题怎么做？」「为什么这里是 90 度？」")
    print("- 也可以描述自己的薄弱知识点，例如：「我对圆柱体积这块比较薄弱，帮我系统讲讲」")
    print("- 当你想多做练习时，可以说：「再来几道类似的题」「给我几道相关练习题」")
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

        # 支持在对话中途更新图片：
        # 方式 1：单独设置图片，不附带问题
        #   :image 路径/to/img.png
        #   image 路径/to/img.png
        #   :img   路径/to/img.png
        #   img    路径/to/img.png
        #   :图片  路径/to/img.png
        #   图片   路径/to/img.png
        # 方式 2：同一轮里既传图片又提问
        #   image 路径/to/img.png 这道题怎么做？
        tokens = user_input.split(maxsplit=2)
        if tokens:
            cmd = tokens[0].lower()
            if cmd in {":image", "image", ":img", "img", ":图片", "图片"}:
                if len(tokens) >= 2:
                    new_img = tokens[1].strip()
                    state.user_image_path = new_img
                    print(f"已更新当前题目图片为: {state.user_image_path}")
                # 如果这一行除了设置图片没有附带问题，就进入下一轮让用户再输入问题
                if len(tokens) < 3:
                    continue
                # 否则把剩余部分当作本轮实际的问题文本
                user_input = tokens[2].strip()
                lower = user_input.lower()
                if not user_input:
                    continue

        action = decide_action(user_input, state)

        # 由“对话管理层”来选择底层工具
        if action == "solve":
            # 解题 / 知识点讲解：走完整双 RAG 流程
            print("\n助手：我先帮你检索相关题目和知识点，再给你详细讲解...")
            result = agent.run(
                user_question=user_input,
                user_image_path=state.user_image_path,
            )
            answer = result.get("answer", "") or ""

            print("\n" + "=" * 60)
            print("【助手回答】")
            print("=" * 60)
            print(answer)

            # 记录这次问题，供后续“类似题”推荐使用
            state.last_question = user_input
            state.last_exercise_seed = user_input

            print("\n（如果你想多做几道类似的题，可以直接说：再来几道类似的题。）")

        elif action in {"recommend_from_last", "recommend_from_text"}:
            # 习题推荐：调用专门的推荐工具
            if action == "recommend_from_last" and state.last_exercise_seed:
                seed = state.last_exercise_seed
            else:
                seed = user_input

            print("\n助手：我根据你当前的问题/薄弱点，帮你找一些相似的练习题...")
            exercises = agent.recommend_exercises(
                user_question=seed,
                user_image_path=state.user_image_path,
                n_exercises=5,
            )
            print_exercises(exercises)

            # 更新最近一次用于练习推荐的种子
            state.last_exercise_seed = seed

            # 记录本次推荐的习题列表，便于后续“第2题怎么做”这种指代
            state.last_exercises = exercises

        elif action == "solve_recommended":
            # 用户指名“第 n 题怎么做”，优先在最近一次推荐的题目列表里找
            import re

            if not state.last_exercises:
                print("目前还没有可用的练习题列表，可以先让我推荐几道相关的题目。")
                continue

            m = re.search(r"第\s*(\d+)\s*题", user_input)
            if not m:
                print("我没有理解你指的是哪一题，可以说：第2题怎么做、详细讲讲第 1 题等。")
                continue

            idx = int(m.group(1)) - 1
            if idx < 0 or idx >= len(state.last_exercises):
                print(f"目前一共只有 {len(state.last_exercises)} 题可选，你说的第 {idx + 1} 题不在范围内。")
                continue

            ex = state.last_exercises[idx]
            ex_question = (ex.get("question") or "").strip()
            ex_kc = (ex.get("knowledge_concept") or "").strip()

            if not ex_question:
                print("这道题在题库中没有找到完整题干，暂时无法单独讲解。你可以换一题或重新描述你的问题。")
                continue

            # 构造一个更明确的“用户问题”，带上题目文本和知识点，交给底层 RAG + LLM 处理
            composed_question = f"请帮我详细讲解这道题，并给出完整的解题步骤和思路：{ex_question}"
            if ex_kc:
                composed_question += f"（相关知识点：{ex_kc}）"

            print(f"\n助手：我来详细讲讲你刚才推荐列表里的第 {idx + 1} 题。")
            result = agent.run(
                user_question=composed_question,
                user_image_path=ex.get("image_path") or state.user_image_path,
            )
            answer = result.get("answer", "") or ""

            print("\n" + "=" * 60)
            print(f"【第 {idx + 1} 题的详细讲解】")
            print("=" * 60)
            print(answer)

            # 更新 last_question，方便后续“类似题”继续围绕这道题来推荐
            state.last_question = composed_question
            state.last_exercise_seed = composed_question


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两阶段多模态 RAG 对话式辅导")
    parser.add_argument("--image", "-i", type=str, default="", help="题目图片路径（可选），解题与推荐练习时会用到")
    args = parser.parse_args()
    interactive_chat(user_image_path=args.image or "")

