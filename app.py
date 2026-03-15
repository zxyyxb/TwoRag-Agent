"""
前端问答后端：提供 Web 页面与 /api/ask 接口，与终端用法一致（问题 + 可选图片 → 回答）
"""
import base64
import os
import tempfile
import uuid

from flask import Flask, request, jsonify, send_from_directory

from config import BASE_DIR
from rag_agent import RAGAgent

app = Flask(__name__, static_folder="static", static_url_path="")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_base64_image(data_url: str) -> str | None:
    """将 data URL (data:image/xxx;base64,...) 保存为临时文件，返回路径。"""
    if not data_url or not data_url.startswith("data:"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
        ext = ".png"
        if "image/jpeg" in header or "image/jpg" in header:
            ext = ".jpg"
        path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as f:
            f.write(raw)
        return path
    except Exception:
        return None


@app.route("/")
def index():
    """返回前端页面"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    接收用户问题和可选图片，调用 RAG Agent 返回回答。
    请求体 JSON: { "question": "这题怎么做", "image_base64": null 或 "data:image/png;base64,..." }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        image_base64 = data.get("image_base64")

        image_path = ""
        if image_base64:
            image_path = _save_base64_image(image_base64) or ""

        agent = RAGAgent()
        result = agent.run(
            user_question=question or "",
            user_image_path=image_path,
        )

        answer = result.get("answer") or ""
        return jsonify({
            "ok": True,
            "answer": answer,
            "targeted_keywords": result.get("targeted_keywords", []),
            "top_results_count": len(result.get("top_results", [])),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "answer": ""}), 500


if __name__ == "__main__":
    print("启动前端服务：http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
