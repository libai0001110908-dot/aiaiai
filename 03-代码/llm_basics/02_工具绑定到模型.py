"""
第 05 章 Tools 练习 2：工具绑定到模型（需要 API Key）
=================================================
本练习需要大模型 API，让 AI 自己决定何时调用工具。
学习目标：
  1. 初始化大模型
  2. 使用 bind_tools() 绑定工具到模型
  3. 让 AI 决定是否调用工具
  4. 完整的工具调用流程（四步骤）
  5. 多工具并行调用

前置条件：
  1. 将 configs/.env.example 复制为 configs/.env
  2. 填入你的 API Key（推荐 DeepSeek，国内可用）

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/llm_basics/02_工具绑定到模型.py"
"""

import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# =========================================================
# 1. 加载环境变量 & 初始化模型
# =========================================================

# 加载 .env 文件（从 configs 目录读取）
load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", ".env"
), override=True)

# 课件中使用 CLOSEAI 命名，这里兼容 OpenAI 和 DeepSeek
# 方式 A：使用 OpenAI 兼容接口（课件写法）
API_KEY = os.getenv("CLOSEAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("CLOSEAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

if not API_KEY:
    print("⚠️  未检测到 API Key！")
    print("请按以下步骤操作：")
    print("  1. 复制 configs/.env.example 为 configs/.env")
    print("  2. 填入你的 API Key（推荐 DeepSeek）")
    print("  3. 重新运行本脚本")
    exit(1)

# 初始化模型（可按需修改 model 名称）
model = init_chat_model(
    model="deepseek-chat",          # DeepSeek 模型
    model_provider="openai",        # 使用 OpenAI 兼容协议
    api_key=API_KEY,
    base_url=BASE_URL,
)

print(f"✅ 模型初始化成功，base_url={BASE_URL}")


# =========================================================
# 2. 定义工具
# =========================================================

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 22°C",
        "广州": "小雨，温度 28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def search_web(query: str) -> str:
    """搜索网页内容"""
    return f"搜索 '{query}' 的结果：\n  1. 相关结果一\n  2. 相关结果二"


# =========================================================
# 3. 绑定工具到模型
# =========================================================

# 工具字典：方便后续通过名称查找并调用
tools = [get_weather, calculate, search_web]
tools_map = {t.name: t for t in tools}

# 绑定工具
model_with_tools = model.bind_tools(tools)

print(f"✅ 已绑定 {len(tools)} 个工具: {list(tools_map.keys())}")


# =========================================================
# 4. 完整的工具调用流程（四步骤）
# =========================================================

print("\n" + "=" * 60)
print("练习 1：完整的工具调用流程")
print("=" * 60)

# 步骤 1：用户提问
user_question = "北京今天天气怎么样？"
print(f"\n👤 用户: {user_question}")

# 步骤 2：模型决定是否调用工具
response = model_with_tools.invoke([HumanMessage(content=user_question)])

if response.tool_calls:
    print(f"🤖 AI 想调用工具: {response.tool_calls}")

    # 步骤 3：执行工具调用
    messages = [HumanMessage(content=user_question), response]

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"   🔧 调用 {tool_name}({tool_args})")

        # 执行工具
        tool_result = tools_map[tool_name].invoke(tool_args)
        print(f"   📦 工具返回: {tool_result}")

        # 步骤 4：将工具结果反馈给模型
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"],
        ))

    # 模型根据工具结果生成最终回复
    final_response = model_with_tools.invoke(messages)
    print(f"\n🤖 AI 最终回复: {final_response.content}")
else:
    print(f"🤖 AI 直接回答: {response.content}")


# =========================================================
# 5. 测试不需要工具的场景
# =========================================================

print("\n" + "=" * 60)
print("练习 2：不需要工具的场景")
print("=" * 60)

user_question2 = "你好，请介绍一下你自己"
print(f"\n👤 用户: {user_question2}")

response2 = model_with_tools.invoke([HumanMessage(content=user_question2)])

if response2.tool_calls:
    print(f"🤖 AI 想调用工具: {response2.tool_calls}")
else:
    print(f"🤖 AI 直接回答: {response2.content}")


# =========================================================
# 6. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
工具调用的完整四步骤：
  步骤 1: model.bind_tools([...])     → 绑定工具
  步骤 2: model.invoke(用户问题)      → 模型返回 AIMessage（含 tool_calls）
  步骤 3: 执行工具，得到结果          → tool.invoke(tool_call)
  步骤 4: 将 ToolMessage 反馈给模型   → 模型生成最终回复

下一步尝试：
  - 03_强制使用工具.py：学习 tool_choice 参数
  - 04_多工具并行调用.py：一次调用多个工具
""")
