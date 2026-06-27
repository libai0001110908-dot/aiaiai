# AI 大模型学习项目

本仓库用于系统化学习 AI 大模型开发，包含课件、笔记、代码练习与实战项目。

## 目录结构

```
aiaiiai/
├── 01-课件/              # 课程 PDF 课件
├── 02-笔记/              # 学习笔记（Markdown）
├── 03-代码/              # 代码示例与练习
│   ├── langchain/        #   LangChain 框架相关代码
│   ├── llm_basics/       #   大模型基础调用与概念
│   ├── rag/              #   检索增强生成（RAG）
│   └── agents/           #   智能体（Agent）相关
├── 04-实战项目/          # 综合实战项目
├── 05-数据集/            # 训练与测试数据
├── 06-模型/              # 模型文件与权重
├── docs/                 # 项目文档
├── notebooks/            # Jupyter Notebook 探索性实验
├── configs/              # 配置文件（模型参数、API Key 等）
├── scripts/              # 工具脚本（数据预处理、部署等）
├── tests/                # 单元测试
├── utils/                # 公共工具函数
├── requirements.txt      # Python 依赖
└── README.md             # 项目说明
```

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/libai0001110908-dot/aiaiai.git
cd aiaiai

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

# 4. 安装依赖
pip install -r 01-课件/requirements.txt
```

## 学习路线

1. **LangChain 概述** — 了解框架整体架构
2. **模型的创建与调用** — 接入大模型 API
3. **LangSmith 的使用** — 链路追踪与调试
4. **Message 与提示词模板** — 提示工程
5. **Tools** — 工具调用
6. **结构化输出** — 输出格式控制
7. **智能体** — Agent 构建与编排
8. **中间件** — 请求处理链
9. **上下文与记忆** — 会话状态管理
10. **RAG** — 检索增强生成

## 提交规范

```bash
git add -A
git commit -m "feat: 新增 RAG 示例代码"
git push
```
