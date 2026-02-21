"""
两阶段多模态 RAG 主入口
用法：
  1. 先运行 build_vectors.py 构建向量库
  2. 再运行本脚本进行检索与问答
"""
import argparse
from rag_agent import RAGAgent



def main():
    parser = argparse.ArgumentParser(description="两阶段多模态 RAG")
    parser.add_argument("--question", "-q", type=str, default="", help="用户问题")
    parser.add_argument("--image", "-i", type=str, default="", help="用户上传的题目图像路径（可选）")
    parser.add_argument("--text-top-k", type=int, default=20, help="文本 RAG 召回数")
    parser.add_argument("--image-top-k", type=int, default=3, help="图像 RAG 精筛 Top-K")
    parser.add_argument("--no-react", action="store_true", help="禁用 ReAct 分步思考日志")
    args = parser.parse_args()

    agent = RAGAgent()

    question = args.question or "正方形顶点为圆心画圆，求圆心角 ∠ECF 的度数"
    result = agent.run(
        user_question=question,
        user_image_path=args.image,
        text_top_k=args.text_top_k,
        image_top_k=args.image_top_k,
        react_mode=not args.no_react,
    )

    print("\n" + "=" * 60)
    print("【最终回答】")
    print("=" * 60)
    print(result["answer"])
    print("\n【靶向词】", result["targeted_keywords"])
    print("\n【Top 结果数量】", len(result["top_results"]))
    if result.get("react_log_path"):
        print("\n【ReAct 日志】", result["react_log_path"])


if __name__ == "__main__":
    main()
