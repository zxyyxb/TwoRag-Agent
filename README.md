# 两阶段多模态 RAG 方案（靶向词后置生成）

## 一、整体流程

```
文本 RAG 粗召回 → 基于召回数据生成靶向词 → 靶向词 + 用户图像做图像 RAG 精筛 Top-3 → 聚合生成回答
```

## 二、项目结构

```
毕业论文/
├── we_math_cleaned.csv          # 数据集（含 question, image_path, knowledge concept, answer 等）
├── extracted_images/            # 题目图像文件夹（与 CSV 中 image_path 对应）
├── config.py                    # 配置参数
├── build_vectors.py             # 离线构建：文本向量化 + 图像向量化
├── rag_agent.py                 # 两阶段 RAG Agent（检索 + 靶向词 + 精筛 + 聚合）
├── answer_generator.py          # LLM 回答生成（贴合用户场景）
├── vector_store.py              # NumPy 向量存储
├── main.py                      # 主入口
├── requirements.txt
├── chroma_db/                   # ChromaDB 持久化目录（运行 build 后生成）
└── README.md
```

## 三、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建向量库（离线阶段，只需执行一次）

```bash
python build_vectors.py
```

- 会读取 `we_math_cleaned.csv`
- 图像路径：`extracted_images/` 下与 CSV 中 `image_path` 文件名对应（如 `1-1.png`）
- 生成 `chroma_db/`，内含文本向量集合与图像向量集合

### 3. 运行检索与问答

```bash
# 仅文本问题
python main.py -q "正方形顶点为圆心画圆，求圆心角度数"

# 文本 + 用户上传的题目图像
python main.py -q "如图，求圆心角" -i "path/to/user_image.png"
```

## 四、配置说明（config.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `IMAGE_BASE_DIR` | 图像根目录 | `extracted_images` |
| `TEXT_TOP_K` | 文本 RAG 召回数 | 20 |
| `IMAGE_TOP_K` | 图像 RAG 精筛 Top-K | 3 |
| `IMAGE_SIM_WEIGHT` | 图像相似度权重 | 0.7 |
| `KEYWORD_SIM_WEIGHT` | 靶向词-图像匹配权重 | 0.3 |
| `USE_LLM_ANSWER` | 是否用 LLM 生成贴合用户场景的回答 | True |
| `LLM_PROVIDER` | LLM 提供商 | openai / ollama |
| `OPENAI_API_KEY` | OpenAI API Key（环境变量） | - |
| `OPENAI_MODEL` | OpenAI 模型名 | gpt-4o-mini |
| `OLLAMA_MODEL` | Ollama 本地模型名 | qwen2.5:7b |

### LLM 回答生成

- **OpenAI**：设置环境变量 `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL`（如代理或本地 API）
- **Ollama**：本地运行 `ollama run qwen2.5:7b`，将 `LLM_PROVIDER` 设为 `ollama`
- 若 `USE_LLM_ANSWER=False` 或无可用 LLM，则仅返回检索结果模板

## 五、流程模块说明

1. **文本 RAG**：对「题目 + 知识点」做文本向量化，用 ChromaDB 余弦检索
2. **靶向词生成**：基于 Top-K 候选的题目、知识点提炼关键词/概念，供图像 RAG 使用
3. **图像 RAG**：用 CLIP 对候选图像与用户图像、靶向词编码，融合相似度重排取 Top-3
4. **答案聚合**：优先用 LLM 生成贴合用户场景的自然语言回答；否则返回检索结果模板
