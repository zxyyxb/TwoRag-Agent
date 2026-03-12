"""
ReAct 模式日志记录器
输出格式：Thought -> Action -> Observation，体现两阶段多模态 RAG 的分步推理。
与 rag_agent 的工具化调用配合：每一步对应一次工具调用（如 text_rag_retrieve、image_rag_refine、aggregate_answer）。
"""
import os
from datetime import datetime


class ReactLogger:
    """ReAct 分步思考日志，写入文件"""

    def __init__(self, log_dir: str = "", enabled: bool = True):
        self.enabled = enabled
        self.log_dir = log_dir or "react_logs"
        self.log_path = ""
        self._lines = []

    def _ensure_dir(self):
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)

    def start(self, user_question: str, user_image: str = ""):
        """开始新一次对话的 ReAct 日志"""
        self._lines = []
        if not self.enabled:
            return
        self._ensure_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"react_{ts}.log")
        self._write(f"{'='*60}")
        self._write(f"ReAct 两阶段多模态 RAG 推理日志")
        self._write(f"{'='*60}")
        self._write(f"时间: {datetime.now().isoformat()}")
        self._write(f"用户问题: {user_question}")
        self._write(f"用户图像: {user_image or '(无)'}")
        self._write("")

    def step(self, step_name: str, thought: str, action: str, observation: str):
        """记录一个 ReAct 步骤：Thought -> Action -> Observation"""
        if not self.enabled:
            return
        self._write(f"--- Step: {step_name} ---")
        self._write("Thought: " + thought)
        self._write("Action: " + action)
        self._write("Observation: " + observation)
        self._write("")

    def final(self, answer: str):
        """记录最终回答"""
        if not self.enabled:
            return
        self._write("--- Final Answer ---")
        self._write(answer[:2000] + ("..." if len(answer) > 2000 else ""))
        self._write("")
        self._write(f"{'='*60}\n")

    def _write(self, s: str):
        self._lines.append(s)
        if self.enabled and self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(s + "\n")

    def flush(self):
        """刷新并关闭"""
        pass
