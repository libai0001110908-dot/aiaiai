"""
第 10 章 RAG 练习 1：文本切分与向量化基础
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 文档加载（Document 对象）
  2. 文本切分器对比：
     - CharacterTextSplitter（按固定字符分割）
     - RecursiveCharacterTextSplitter（递归分割，最常用）
     - TokenTextSplitter（按 token 分割）
  3. 切分参数详解（chunk_size, chunk_overlap）
  4. 自定义分隔符（中文场景）
  5. 向量化概念理解

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/rag/01_文本切分与向量化.py"
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


# =========================================================
# 1. Document 对象
# =========================================================

print("=" * 60)
print("练习 1：Document 对象")
print("=" * 60)

# Document 是 LangChain 中文档的基本单位
doc = Document(
    page_content="LangChain 是一个用于开发大语言模型应用的开源框架。",
    metadata={"source": "课件", "chapter": "01", "author": "尚硅谷"},
)

print(f"page_content: {doc.page_content}")
print(f"metadata: {doc.metadata}")
print(f"类型: {type(doc)}")

# 可以创建多个 Document
docs = [
    Document(page_content="这是第一段内容。", metadata={"section": 1}),
    Document(page_content="这是第二段内容。", metadata={"section": 2}),
]
print(f"\n文档数量: {len(docs)}")


# =========================================================
# 2. CharacterTextSplitter（按固定字符分割）
# =========================================================

print("\n" + "=" * 60)
print("练习 2：CharacterTextSplitter（按字符分割）")
print("=" * 60)

text = """
人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
"""

# CharacterTextSplitter：按单一分隔符分割
char_splitter = CharacterTextSplitter(
    separator="\n\n",    # 分隔符
    chunk_size=50,       # 每块最大字符数
    chunk_overlap=10,    # 块之间的重叠字符数
)

char_chunks = char_splitter.split_text(text)
print(f"分隔符: '\\n\\n'")
print(f"chunk_size=50, chunk_overlap=10")
print(f"切分结果: {len(char_chunks)} 块")
for i, chunk in enumerate(char_chunks):
    print(f"\n--- 块 {i+1} (长度:{len(chunk)}) ---")
    print(f"{chunk.strip()}")


# =========================================================
# 3. RecursiveCharacterTextSplitter（递归分割，最常用）
# =========================================================

print("\n" + "=" * 60)
print("练习 3：RecursiveCharacterTextSplitter（递归分割，最常用）")
print("=" * 60)

# RecursiveCharacterTextSplitter：按分隔符列表递归尝试
# 优先按 \n\n 分，不够小再按 \n 分，再按空格分，最后按字符分
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,          # 每块最大字符数
    chunk_overlap=20,       # 块之间的重叠
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分隔符优先级
)

recursive_chunks = recursive_splitter.split_text(text)
print(f"chunk_size=80, chunk_overlap=20")
print(f"分隔符优先级: ['\\n\\n', '\\n', '。', '，', ' ', '']")
print(f"切分结果: {len(recursive_chunks)} 块")
for i, chunk in enumerate(recursive_chunks):
    print(f"\n--- 块 {i+1} (长度:{len(chunk)}) ---")
    print(f"{chunk.strip()}")


# =========================================================
# 4. chunk_size 和 chunk_overlap 详解
# =========================================================

print("\n" + "=" * 60)
print("练习 4：chunk_size 和 chunk_overlap 详解")
print("=" * 60)

sample_text = "人工智能是计算机科学的分支。" * 10  # 约 130 字符

# 不同 chunk_size 的对比
for size in [30, 50, 100]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=0,  # 先看不重叠的情况
    )
    chunks = splitter.split_text(sample_text)
    print(f"\nchunk_size={size}, overlap=0 → {len(chunks)} 块")
    for i, c in enumerate(chunks[:3]):  # 只显示前3块
        print(f"  块{i+1} (长度:{len(c)}): {c[:40]}...")

# 不同 overlap 的对比
print("\n--- chunk_overlap 对比 ---")
for overlap in [0, 10, 20]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(sample_text)
    print(f"  chunk_size=50, overlap={overlap} → {len(chunks)} 块")

print("""
chunk_size：每块的最大长度
  - 太小：语义不完整，上下文丢失
  - 太大：检索精度下降，token 消耗大
  - 建议：中文 200-500 字符，英文 500-1000 字符

chunk_overlap：相邻块之间的重叠部分
  - 作用：防止在切分点丢失上下文
  - 建议：chunk_size 的 10%-20%
""")


# =========================================================
# 5. 中文场景的自定义分隔符
# =========================================================

print("=" * 60)
print("练习 5：中文场景的自定义分隔符")
print("=" * 60)

chinese_text = """
第一章 LangChain概述。LangChain是一个用于开发大语言模型应用的开源框架。它提供了模型调用、工具使用、智能体构建等核心功能。

第二章 模型的创建与调用。LangChain支持多种大模型供应商，包括OpenAI、DeepSeek、Anthropic等。使用init_chat_model方法可以统一初始化。

第三章 消息与提示词模板。LangChain定义了四种消息类型：System、Human、AI和Tool。ChatPromptTemplate是推荐的提示词模板。
"""

# 中文场景：优先按段落分，再按句号分
chinese_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)

chinese_chunks = chinese_splitter.split_text(chinese_text)
print(f"中文切分结果: {len(chinese_chunks)} 块")
for i, chunk in enumerate(chinese_chunks):
    print(f"\n--- 块 {i+1} (长度:{len(chunk)}) ---")
    print(f"{chunk.strip()}")


# =========================================================
# 6. 使用 create_documents 和 split_documents
# =========================================================

print("\n" + "=" * 60)
print("练习 6：create_documents 和 split_documents")
print("=" * 60)

# create_documents：从字符串列表创建 Document
texts = [
    "第一篇文档的内容。这里有一些关于AI的介绍。",
    "第二篇文档的内容。这里是关于机器学习的讨论。",
]
docs_from_text = recursive_splitter.create_documents(texts)
print(f"create_documents 结果: {len(docs_from_text)} 个 Document")
for i, doc in enumerate(docs_from_text):
    print(f"  Doc {i+1}: {doc.page_content[:50]}...")

# split_documents：对已有的 Document 列表进行切分
original_docs = [
    Document(page_content=chinese_text, metadata={"source": "课件"}),
]
split_docs = recursive_splitter.split_documents(original_docs)
print(f"\nsplit_documents 结果: {len(split_docs)} 个 Document")
for i, doc in enumerate(split_docs):
    print(f"  Doc {i+1} (metadata={doc.metadata}): {doc.page_content[:40]}...")


# =========================================================
# 7. TokenTextSplitter（按 token 分割）
# =========================================================

print("\n" + "=" * 60)
print("练习 7：TokenTextSplitter（按 token 分割）")
print("=" * 60)

try:
    token_splitter = TokenTextSplitter(
        chunk_size=50,       # 每块最大 token 数
        chunk_overlap=10,    # 重叠 token 数
    )
    token_chunks = token_splitter.split_text(text)
    print(f"chunk_size=50 tokens, overlap=10 tokens")
    print(f"切分结果: {len(token_chunks)} 块")
    for i, chunk in enumerate(token_chunks[:3]):
        print(f"\n--- 块 {i+1} ---")
        print(f"{chunk.strip()}")
except Exception as e:
    print(f"TokenTextSplitter 需要额外依赖: {e}")
    print("（需要安装 tiktoken 库）")


# =========================================================
# 8. 向量化概念
# =========================================================

print("\n" + "=" * 60)
print("练习 8：向量化概念")
print("=" * 60)

print("""
向量化（Embedding）是 RAG 的核心步骤：

  文本 → 嵌入模型 → 向量（高维数字数组）→ 向量数据库

  "人工智能" → [0.12, -0.34, 0.56, ..., 0.78]  (768维或1536维)

向量化的意义：
  语义相近的文本，向量距离也相近
  → 可以通过数学计算（余弦相似度）找到"意思最接近"的文本

RAG 完整流程：
  ┌──────────────────────────────────────────────┐
  │  离线阶段（构建知识库）                        │
  │  文档 → 加载 → 切分 → 向量化 → 存入向量数据库  │
  ├──────────────────────────────────────────────┤
  │  在线阶段（检索问答）                          │
  │  用户问题 → 向量化 → 检索相似文档 → 拼接上下文  │
  │  → 传给大模型 → 生成回答                       │
  └──────────────────────────────────────────────┘

常用嵌入模型：
  ┌────────────────────┬──────────────────────────┐
  │  模型              │  说明                    │
  ├────────────────────┼──────────────────────────┤
  │  text-embedding-3  │  OpenAI 嵌入模型         │
  │  bge-large-zh      │  智源研究院，中文优秀     │
  │  m3e-base          │  开源中文嵌入模型        │
  └────────────────────┴──────────────────────────┘

常用向量数据库：
  ┌────────────────────┬──────────────────────────┐
  │  数据库            │  说明                    │
  ├────────────────────┼──────────────────────────┤
  │  FAISS             │  Meta开源，本地使用       │
  │  Milvus            │  企业级，支持分布式       │
  │  Chroma            │  轻量级，开发友好         │
  └────────────────────┴──────────────────────────┘
""")


# =========================================================
# 9. 切分器对比总结
# =========================================================

print("=" * 60)
print("练习 9：切分器对比总结")
print("=" * 60)

comparison = """
┌──────────────────────────┬────────────┬──────────────────────────────┐
│  切分器                   │  分割依据   │  特点                        │
├──────────────────────────┼────────────┼──────────────────────────────┤
│  CharacterTextSplitter   │  固定分隔符 │  简单，按单一分隔符切分       │
│  RecursiveCharacterText  │  递归分隔符 │  最常用，多级分隔符回退      │
│  TokenTextSplitter       │  token 数量 │  精确控制 token，适合大模型  │
│  SemanticChunker         │  语义相似度 │  智能分割，但需要嵌入模型    │
└──────────────────────────┴────────────┴──────────────────────────────┘

选型建议：
  - 通用场景：RecursiveCharacterTextSplitter（最常用）
  - 中文场景：自定义分隔符 ["\\n\\n", "\\n", "。", "，", " ", ""]
  - 精确控制：TokenTextSplitter
  - 高质量切分：SemanticChunker（需嵌入模型）
"""

print(comparison)

# =========================================================
# 10. 练习小结
# =========================================================

print("=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 10 章 RAG 的文本处理基础：

  1. Document 对象：LangChain 文档的基本单位
  2. CharacterTextSplitter：按固定分隔符分割
  3. RecursiveCharacterTextSplitter：递归分割（最常用）
  4. chunk_size / chunk_overlap：控制块大小和重叠
  5. 中文场景：自定义分隔符优先级
  6. create_documents / split_documents：文档操作方法
  7. TokenTextSplitter：按 token 数量分割
  8. 向量化概念：文本 → 向量 → 检索

RAG 核心流程：
  离线：文档 → 加载 → 切分 → 向量化 → 向量数据库
  在线：问题 → 向量化 → 检索 → 拼接上下文 → 大模型 → 回答

下一步：
  - 配置 API Key 后，运行完整 RAG 流程（向量化 + 检索 + 生成）
  - 安装 RAG 章节依赖（milvus, pymilvus, transformers 等）
""")
