"""
LLM 回答生成模块：基于检索结果生成贴合用户场景的自然语言回答
支持 OpenAI API 和 Ollama 本地模型；传入用户图片时使用视觉模型识别题目
"""
import base64
import os

from config import (
    USE_LLM_ANSWER,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    OPENAI_MAX_TOKENS,
    OLLAMA_MODEL,
    BASE_DIR,
    IMAGE_BASE_DIR,
)


def _build_context(top_results: list, targeted_keywords: list) -> str:
    """将检索结果整理为 LLM 可用的上下文"""
    lines = []
    for i, r in enumerate(top_results, 1):
        md = r.get("metadata", {})
        q = md.get("question", "")
        kc = md.get("knowledge_concept", "")
        ans = md.get("answer", "")
        opt = md.get("option", "")
        desc = md.get("knowledge_concept_desc", "")[:500]
        lines.append(f"[题目{i}]\n题目: {q}\n知识点: {kc}\n选项: {opt}\n答案: {ans}\n知识点描述: {desc}\n")
    lines.append(f"\n靶向关键词: {', '.join(targeted_keywords)}")
    return "\n".join(lines)


def _resolve_image_path(image_path: str) -> str | None:
    """解析图片实际路径，支持多种写法"""
    if not image_path or not image_path.strip():
        return None
    image_path = image_path.strip()
    candidates = []
    if os.path.isabs(image_path):
        candidates.append(image_path)
    else:
        base_name = os.path.basename(image_path)
        candidates.append(os.path.join(IMAGE_BASE_DIR, base_name))
        candidates.append(os.path.join(BASE_DIR, image_path))
        candidates.append(os.path.join(BASE_DIR, base_name))
        candidates.append(image_path)  # 相对当前工作目录
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def _image_to_base64(image_path: str) -> str | None:
    """将图片转为 base64 data URL，供视觉模型使用"""
    path = _resolve_image_path(image_path)
    if not path:
        print(f"[提示] 未找到图片: {image_path}，LLM 将无法查看图像。请确认路径正确。")
        return None
    try:
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower() or ".png"
        mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[提示] 读取图片失败: {path}, {e}")
        return None


def _call_openai(user_question: str, context: str, user_image_path: str = "") -> str:
    """调用 OpenAI 兼容 API，支持视觉模型（传入图片时）"""
    try:
        from openai import OpenAI
    except ImportError:
        return ""

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    prompt = f"""你是一个数学题目辅导助手。用户提问并附带了题目图像，你已经检索到以下相似题目与知识点作为参考。

【用户问题】
{user_question}

【检索到的参考题目与知识点】
{context}

请先仔细观察用户上传的题目图像，再结合上述检索结果，贴合用户问题，用自然、友好的语言生成一份回答。

**严格要求（无论是否检索到参考题目，均须遵守）：**
- 所有数值（直径、半径、高、长度等）必须只使用你在图中或题目中实际给出的数据，不得编造或假设。
- 若图中或题目中数值模糊、未给出或缺少必要条件（如角度、某条边长、比例等），必须明确写出「无法从图中/题目中读取」或「根据当前条件无法唯一求解」，**不得假设或编造缺失数据再作答**。
- 若能从图中或题目中确认具体数值，请写出你看到的实际数值并据此计算。

要求：
1. 结合图像内容，概括题目涉及的知识点；
2. 根据你从图中读取到的实际数据，给出解题思路或答案；
3. 语言简洁、条理清晰，适合学生理解；
4. 回答必须完整收尾：给出最终答案或结论，不要写到一半中断。
5. **多种情况必须穷举**：若题目存在多种情况、多种分类或多种取法，必须逐一讨论并给出每种情况的结果，最后汇总所有答案，不要只算一种就结束。
"""
    image_url = _image_to_base64(user_image_path)
    if image_url:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        model = OPENAI_VISION_MODEL  # 传图时用视觉模型
        print(f"[已加载图像] 使用视觉模型 {model} 识别题目图")
    else:
        content = prompt
        model = OPENAI_MODEL

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM 调用失败: {e}]"


def _call_ollama(user_question: str, context: str) -> str:
    """调用 Ollama 本地模型"""
    try:
        import requests
    except ImportError:
        return ""

    prompt = f"""你是一个数学题目辅导助手。用户提问并附带了相关题目图像，你已经检索到以下相似题目与知识点作为参考。

【用户问题】
{user_question}

【检索到的参考题目与知识点】
{context}

请基于上述内容，贴合用户问题，用自然、友好的语言生成一份回答。要求：
1. 先简要概括题目涉及的知识点；
2. 若有参考题目可结合给出解题思路或答案提示；若无参考题目则仅根据用户题目与图像作答；
3. 若可直接作答，给出答案并简要说明；
4. **无论是否检索到参考题目，均不得编造题目或图中未给出的条件与数值**；若条件不足无法求解，必须明确写出「无法确定」或「需补充条件」；
5. 语言简洁、条理清晰，适合学生理解。
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("response", "") or "").strip()
    except Exception as e:
        return f"[Ollama 调用失败: {e}]"


def generate_answer(user_question: str, top_results: list, targeted_keywords: list, user_image_path: str = "") -> str:
    """
    基于检索结果生成贴合用户场景的回答。无论是否检索到相关题目都会调用大模型，
    通过提示词约束模型不编造条件；无检索结果时上下文会提示「仅根据用户题目与图像作答」。
    """
    if not USE_LLM_ANSWER:
        return ""

    if top_results:
        context = _build_context(top_results, targeted_keywords)
    else:
        context = (
            "【注意】未检索到相关题目。请仅根据用户问题与题目图像作答，"
            "切勿编造题目中未给出的条件或数值；若条件不足无法求解，请明确说明需补充的条件。"
        )

    if LLM_PROVIDER == "ollama":
        return _call_ollama(user_question, context)
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            print("[提示] 未设置 OPENAI_API_KEY，使用检索结果模板。请在运行前执行: $env:OPENAI_API_KEY = \"sk-xxx\"")
            return ""
        result = _call_openai(user_question, context, user_image_path)
        if not result:
            print("[提示] LLM 返回空内容，请检查模型名或 API 配置。")
        return result
    return ""
