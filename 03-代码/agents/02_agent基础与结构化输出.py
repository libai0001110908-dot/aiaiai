"""
第 07 章 智能体 练习 2：Agent 基础与结构化输出
=================================================
本练习需要大模型 API，创建真正的 Agent 并调用。
学习目标：
  1. create_agent() 创建 Agent
  2. Agent 绑定工具并自动调用
  3. Agent 的多轮工具调用
  4. response_format 结构化输出（ProviderStrategy / ToolStrategy）
  5. Agent 命名与提示词设置
  6. 流式输出（stream）

前置条件：
  1. 将 configs/.env.example 复制为 configs/.env
  2. 填入你的 API Key（推荐 DeepSeek）

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/agents/02_agent基础与结构化输出.py"
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool


# =========================================================
# 1. 加载环境变量 & 初始化模型
# =========================================================

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", ".env"
), override=True)

API_KEY = os.getenv("CLOSEAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("CLOSEAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

if not API_KEY:
    print("⚠️  未检测到 API Key！")
    print("请按以下步骤操作：")
    print("  1. 复制 configs/.env.example 为 configs/.env")
    print("  2. 填入你的 API Key（推荐 DeepSeek）")
    print("  3. 重新运行本脚本")
    exit(1)

model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
)
print("✅ 模型初始化成功")


# =========================================================
# 2. 定义工具
# =========================================================

@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息"""
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 22°C",
        "广州": "小雨，温度 28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 "2+3"、"sqrt(144)"、"sin(pi/2)" """
    safe_dict = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_time_info() -> str:
    """获取当前时间信息"""
    now = datetime.now()
    return now.strftime("当前时间：%Y年%m月%d日 %H:%M:%S")


tools = [get_weather, calculate, get_time_info]
print(f"✅ 已准备 {len(tools)} 个工具")


# =========================================================
# 3. 创建 Agent
# =========================================================

print("\n" + "=" * 60)
print("练习 1：创建 Agent 并调用")
print("=" * 60)

agent = create_agent(
    model=model,
    tools=tools,
    prompt="你是一个智能助手，可以查询天气、进行数学计算、查询时间。请用简洁的中文回答。",
    name="smart_assistant",
)

print(f"✅ Agent 创建成功: {agent.name if hasattr(agent, 'name') else 'smart_assistant'}")

# 基本调用
response = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
})

# 获取最后一条 AI 消息
last_msg = response["messages"][-1]
print(f"\n👤 用户: 北京天气怎么样？")
print(f"🤖 Agent: {last_msg.content}")

# 查看完整的消息流转
print(f"\n📊 消息流转（{len(response['messages'])} 条）:")
for i, msg in enumerate(response["messages"]):
    msg_type = type(msg).__name__
    content = str(msg.content)[:80]
    print(f"  [{i}] {msg_type}: {content}{'...' if len(str(msg.content)) > 80 else ''}")


# =========================================================
# 4. 多轮工具调用
# =========================================================

print("\n" + "=" * 60)
print("练习 2：多轮工具调用")
print("=" * 60)

# Agent 会在一次对话中多次调用不同工具
response2 = agent.invoke({
    "messages": [{"role": "user", "content": "帮我查一下北京和上海的天气，然后算一下两地温度差是多少？"}]
})

last_msg2 = response2["messages"][-1]
print(f"👤 用户: 帮我查一下北京和上海的天气，然后算一下两地温度差是多少？")
print(f"🤖 Agent: {last_msg2.content}")

print(f"\n📊 工具调用记录:")
for msg in response2["messages"]:
    msg_type = type(msg).__name__
    if msg_type == "AIMessage" and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  🔧 调用工具: {tc['name']}({tc['args']})")
    elif msg_type == "ToolMessage":
        print(f"  📦 工具返回: {str(msg.content)[:60]}")


# =========================================================
# 5. 结构化输出：ProviderStrategy
# =========================================================

print("\n" + "=" * 60)
print("练习 3：Agent 结构化输出 - ProviderStrategy")
print("=" * 60)

from langchain.agents.structured_output import ProviderStrategy


class WeatherReport(BaseModel):
    """天气报告结构"""
    city: str = Field(description="城市名称")
    weather: str = Field(description="天气状况")
    temperature: str = Field(description="温度")
    suggestion: Optional[str] = Field(default=None, description="出行建议")


# 使用 ProviderStrategy（模型原生结构化输出）
agent_structured = create_agent(
    model=model,
    tools=[get_weather],
    prompt="你是一个天气助手，查询天气后给出结构化报告。",
    response_format=ProviderStrategy(WeatherReport),
)

response3 = agent_structured.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？给我一个结构化报告。"}]
})

# 获取结构化输出
structured_result = response3["structured_response"]
print(f"返回类型: {type(structured_result)}")
print(f"城市: {structured_result.city}")
print(f"天气: {structured_result.weather}")
print(f"温度: {structured_result.temperature}")
print(f"建议: {structured_result.suggestion}")


# =========================================================
# 6. 结构化输出：ToolStrategy
# =========================================================

print("\n" + "=" * 60)
print("练习 4：Agent 结构化输出 - ToolStrategy")
print("=" * 60)

from langchain.agents.structured_output import ToolStrategy


class TimeReport(BaseModel):
    """时间报告结构"""
    date: str = Field(description="日期")
    time: str = Field(description="时间")
    weekday: str = Field(description="星期几")
    greeting: Literal["早上好", "下午好", "晚上好"] = Field(description="问候语")


# 使用 ToolStrategy（通过工具调用实现结构化输出，兼容性更好）
agent_time = create_agent(
    model=model,
    tools=[get_time_info],
    prompt="你是时间助手，查询时间后给出结构化报告。",
    response_format=ToolStrategy(TimeReport),
)

response4 = agent_time.invoke({
    "messages": [{"role": "user", "content": "现在几点了？给我一个时间报告。"}]
})

structured_result4 = response4["structured_response"]
print(f"返回类型: {type(structured_result4)}")
print(f"日期: {structured_result4.date}")
print(f"时间: {structured_result4.time}")
print(f"星期: {structured_result4.weekday}")
print(f"问候: {structured_result4.greeting}")


# =========================================================
# 7. 流式输出
# =========================================================

print("\n" + "=" * 60)
print("练习 5：流式输出（stream）")
print("=" * 60)

print("👤 用户: 北京天气怎么样？")
print("🤖 Agent (流式): ")

# 使用 stream 方法，stream_mode="updates" 可以看到每个步骤
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
    stream_mode="updates",
):
    for node_name, node_output in chunk.items():
        if "messages" in node_output:
            for msg in node_output["messages"]:
                msg_type = type(msg).__name__
                if msg_type == "AIMessage":
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  🔧 [stream] 调用工具: {tc['name']}({tc['args']})")
                    else:
                        print(f"  🤖 [stream] 回复: {msg.content}")
                elif msg_type == "ToolMessage":
                    print(f"  📦 [stream] 工具返回: {msg.content}")


# =========================================================
# 8. 多种流式模式
# =========================================================

print("\n" + "=" * 60)
print("练习 6：多种流式输出模式")
print("=" * 60)

# 同时使用 values 和 updates 两种模式
print("使用 values + updates 双模式:")
for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "sin(pi/2) 等于多少？"}]},
    stream_mode=["values", "updates"],
):
    if stream_mode == "values":
        # values: 每步输出完整状态
        msg_count = len(chunk.get("messages", []))
        print(f"  [values] 当前消息数: {msg_count}")
    elif stream_mode == "updates":
        # updates: 只输出增量变化
        for node, output in chunk.items():
            print(f"  [updates] 节点 '{node}' 更新")


# =========================================================
# 9. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 07 章的核心知识点：

  1. create_agent()：创建 Agent（model + tools + prompt + name）
  2. agent.invoke()：调用 Agent，传入 messages 列表
  3. 多轮工具调用：Agent 自动决定调用顺序和次数
  4. 结构化输出：
     - ProviderStrategy：模型原生结构化输出（推荐）
     - ToolStrategy：通过工具调用实现（兼容性好）
  5. 流式输出 stream()：
     - values：每步输出完整状态
     - updates：只输出增量变化（默认）
     - tasks：任务级输出
     - checkpoints：检查点输出
     - custom：自定义输出

Agent 本质：
  Agent = LangGraph 的 CompiledStateGraph 实例
  底层是图结构，编排"模型推理 → 工具调用 → 结果反馈"的循环

下一步：运行 03_综合实战_智能助手.py，体验完整的多功能 Agent！
""")
