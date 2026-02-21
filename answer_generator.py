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

**严格要求：**
- 所有数值（直径、半径、高、长度等）必须只使用你在图中实际清晰读到的数据，不得编造或假设。
- 若图中数值模糊无法辨认，请明确写出「无法从图中清晰读取具体数值」，并给出解题公式和步骤说明，不要虚构示例数据。
- 若能从图中确认具体数值，请写出你看到的实际数值并据此计算。

要求：
1. 结合图像内容，概括题目涉及的知识点；
2. 根据你从图中读取到的实际数据，给出解题思路或答案；
3. 语言简洁、条理清晰，适合学生理解。
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
            max_tokens=800,
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

请基于上述检索结果，贴合用户问题，用自然、友好的语言生成一份回答。要求：
1. 先简要概括题目涉及的知识点；
2. 结合检索到的相似题目，给出解题思路或答案提示；
3. 若可直接作答，给出答案并简要说明；
4. 语言简洁、条理清晰，适合学生理解。
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
    基于检索结果生成贴合用户场景的回答。
    若 USE_LLM_ANSWER 为 False 或 LLM 不可用，返回空字符串，由调用方用模板兜底。
    """
    if not USE_LLM_ANSWER or not top_results:
        return ""

    context = _build_context(top_results, targeted_keywords)

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
