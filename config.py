"""配置参数"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "we_math_cleaned.csv")
IMAGE_BASE_DIR = os.path.join(BASE_DIR, "extracted_images")  # 图像文件夹，与CSV中image_path拼接
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# ChromaDB 集合名
TEXT_COLLECTION_NAME = "math_text_vectors"
IMAGE_COLLECTION_NAME = "math_image_vectors"

# 检索参数
TEXT_TOP_K = 20          # 第一阶段文本 RAG 召回数量
IMAGE_TOP_K = 3          # 第二阶段图像 RAG 精筛 Top-K
IMAGE_SIM_WEIGHT = 0.7   # 图像相似度权重
KEYWORD_SIM_WEIGHT = 0.3 # 靶向词-图像语义匹配度权重

# 可靠性阈值：超过则视为「条件不足/不可靠」，直接返回无解而非调用 LLM 编造
TEXT_RAG_MAX_DISTANCE = 0.7   # 文本 RAG 最佳匹配允许的最大距离（1-相似度），超过则不可靠
IMAGE_RAG_MIN_SCORE = 0.25    # 图像 RAG 最佳匹配的最低相似度，低于则不可靠

# 模型配置
TEXT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 中英双语
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"  # openai 预训练权重

# LLM 回答生成（贴合用户场景，生成自然语言回答）
USE_LLM_ANSWER = True   # 是否使用 LLM 生成回答，False 则仅返回检索结果
LLM_PROVIDER = "openai"  # openai / ollama（通义千问用 openai，配合下方 base_url）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # 从环境变量读取，勿写死
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # 通义千问
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "qwen-turbo")  # 纯文本
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "qwen-vl-plus")  # 传图时用视觉模型
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))  # 回答最大 token，避免答案被截断
OLLAMA_MODEL = "qwen2.5:7b"   # 本地 Ollama 模型

# ReAct 模式（分步思考日志 + 工具化调用）
REACT_MODE = True             # 是否启用 ReAct 模式，输出分步思考到文件
REACT_LOG_DIR = os.path.join(BASE_DIR, "react_logs")  # 日志输出目录
REACT_TOOL_ORDER = "fixed"    # fixed=固定顺序调用工具；后续可扩展 llm=由 LLM 选择下一工具
