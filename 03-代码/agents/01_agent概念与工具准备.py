"""
第 07 章 智能体 练习 1：Agent 概念与工具准备
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 理解 Agent 的核心要素
  2. 为综合实战准备一组完整的工具
  3. 理解 create_agent 的参数结构
  4. 模拟 Agent 的工具调用流程

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/agents/01_agent概念与工具准备.py"
"""

import json
import math
from datetime import datetime
from langchain_core.tools import tool


# =========================================================
# 1. 理解 Agent 的核心要素
# =========================================================

print("=" * 60)
print("练习 1：Agent 的核心要素")
print("=" * 60)

print("""
Agent（智能体）= 大模型 + 工具 + 推理循环

核心要素：
  ✅ 行动（Action）  — 必须的，Agent 执行操作的能力
  ✅ 工具（Tool）    — 几乎总是存在，让 Agent 与外部世界交互
  🔲 感知（Perception）— 接收环境信息
  🔲 记忆（Memory）  — 保存对话历史和状态
  🔲 规划（Planning）— 分解任务、制定执行计划

一句话总结：Agent = LLM 调用 + 工具调用循环

create_agent 的核心参数：
  model    — Agent 使用的大模型
  tools    — Agent 可调用的工具列表
  prompt   — 系统提示词（定义 Agent 的角色和行为）
  name     — Agent 的名称（多 Agent 协作时区分来源）
""")


# =========================================================
# 2. 为综合实战准备工具集
# =========================================================

print("=" * 60)
print("练习 2：准备工具集（综合实战用）")
print("=" * 60)


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息

    Args:
        city: 城市名称，如 "北京"、"上海"、"广州"

    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 15°C，湿度 40%",
        "上海": "多云，温度 22°C，湿度 65%",
        "广州": "小雨，温度 28°C，湿度 80%",
        "深圳": "阴天，温度 26°C，湿度 70%",
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式，支持加减乘除、三角函数、对数等

    Args:
        expression: 数学表达式，如 "2 + 3"、"sin(3.14/2)"

    Returns:
        计算结果字符串
    """
    # 安全的数学运算环境
    safe_dict = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10,
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "abs": abs, "pow": pow, "round": round,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
    """
    货币汇率转换

    Args:
        amount: 金额
        from_curr: 源货币代码（CNY/USD/EUR/GBP/JPY/HKD）
        to_curr: 目标货币代码（CNY/USD/EUR/GBP/JPY/HKD）

    Returns:
        转换结果字符串
    """
    # 模拟汇率（相对于 CNY）
    rates = {"CNY": 1, "USD": 7.2, "EUR": 7.8, "GBP": 9.1, "JPY": 0.048, "HKD": 0.92}
    from_rate = rates.get(from_curr)
    to_rate = rates.get(to_curr)
    if not from_rate or not to_rate:
        return f"不支持的货币代码，支持：{list(rates.keys())}"
    result = amount * from_rate / to_rate
    return f"{amount} {from_curr} = {result:.2f} {to_curr}"


@tool
def get_time_info(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前时间信息

    Args:
        timezone: 时区，默认 "Asia/Shanghai"

    Returns:
        当前时间信息字符串
    """
    now = datetime.now()
    return (
        f"当前时间（{timezone}）：{now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期：{['一','二','三','四','五','六','日'][now.weekday()]}\n"
        f"今年第 {now.timetuple().tm_yday} 天"
    )


# 查看所有工具
tools = [get_weather, calculate, convert_currency, get_time_info]
print(f"已准备 {len(tools)} 个工具:")
for t in tools:
    print(f"  - {t.name}: {t.description.split(chr(10))[0].strip()}")


# =========================================================
# 3. 测试工具直接调用
# =========================================================

print("\n" + "=" * 60)
print("练习 3：测试工具直接调用")
print("=" * 60)

# 天气查询
print(f"🌤  天气查询:")
print(f"  {get_weather.invoke({'city': '北京'})}")
print(f"  {get_weather.invoke({'city': '上海'})}")

# 数学计算
print(f"\n🔢  数学计算:")
print(f"  {calculate.invoke({'expression': '2 + 3 * 4'})}")
print(f"  {calculate.invoke({'expression': 'sqrt(144)'})}")
print(f"  {calculate.invoke({'expression': 'sin(pi/2)'})}")
print(f"  {calculate.invoke({'expression': 'log(e)'})}")

# 货币转换
print(f"\n💱  货币转换:")
print(f"  {convert_currency.invoke({'amount': 100, 'from_curr': 'USD', 'to_curr': 'CNY'})}")
print(f"  {convert_currency.invoke({'amount': 5000, 'from_curr': 'CNY', 'to_curr': 'JPY'})}")

# 时间查询
print(f"\n⏰  时间查询:")
print(f"  {get_time_info.invoke({})}")


# =========================================================
# 4. 查看工具的 Schema（Agent 通过它理解工具）
# =========================================================

print("\n" + "=" * 60)
print("练习 4：查看工具 Schema")
print("=" * 60)

print("convert_currency 的完整 schema:")
print(json.dumps(
    convert_currency.tool_call_schema.model_json_schema(),
    ensure_ascii=False, indent=2
))


# =========================================================
# 5. 模拟 Agent 的工具调用流程
# =========================================================

print("\n" + "=" * 60)
print("练习 5：模拟 Agent 工具调用流程")
print("=" * 60)

print("""
Agent 工具调用流程（以"北京天气如何？"为例）：

  ┌─────────────────────────────────────────────────────┐
  │  1. 用户输入："北京天气如何？"                        │
  │     → HumanMessage(content="北京天气如何？")          │
  ├─────────────────────────────────────────────────────┤
  │  2. Agent 推理：需要调用天气查询工具                   │
  │     → AIMessage(tool_calls=[{                        │
  │         "name": "get_weather",                       │
  │         "args": {"city": "北京"},                    │
  │         "id": "call_001"                             │
  │       }])                                            │
  ├─────────────────────────────────────────────────────┤
  │  3. 执行工具调用                                      │
  │     → ToolMessage(content="晴天，温度15°C",           │
  │         tool_call_id="call_001")                     │
  ├─────────────────────────────────────────────────────┤
  │  4. Agent 根据工具结果生成最终回复                     │
  │     → AIMessage(content="北京目前晴天，温度15°C...")   │
  └─────────────────────────────────────────────────────┘

  多次工具调用场景（如"北京和上海的天气对比"）：
  Agent 可能分两次调用 get_weather，然后综合结果回答。
""")


# =========================================================
# 6. create_agent 参数结构说明
# =========================================================

print("=" * 60)
print("练习 6：create_agent 参数结构")
print("=" * 60)

# 展示 create_agent 的典型用法（伪代码，需 API Key 才能运行）
print("""
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# 创建 Agent
agent = create_agent(
    model=model,                    # 大模型
    tools=[get_weather, calculate,  # 工具列表（建议 2-5 个）
           convert_currency, get_time_info],
    prompt="你是一个智能助手...",     # 系统提示词
    name="smart_assistant",         # Agent 名称
)

# 调用 Agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
})

# Agent 本质上是 LangGraph 的 CompiledStateGraph 实例
# 底层是一个图结构，编排"模型推理 → 工具调用 → 结果反馈"的循环

注意事项：
  1. 只给 Agent 需要的工具，工具太多会混淆（一般 2-5 个最佳）
  2. 工具的 docstring 要清晰，Agent 通过它理解工具用途
  3. 每次调用必须传递完整的 messages 列表
""")


# =========================================================
# 7. 练习小结
# =========================================================

print("=" * 60)
print("练习小结")
print("=" * 60)
print(f"""
本练习为第 07 章综合实战准备了 {len(tools)} 个工具：
  🌤  get_weather       — 天气查询
  🔢  calculate          — 数学计算
  💱  convert_currency   — 货币转换
  ⏰  get_time_info      — 时间查询

下一步：
  - 运行 02_agent基础与结构化输出.py（需 API Key）
    学习 create_agent、工具绑定、结构化输出
  - 运行 03_综合实战_智能助手.py（需 API Key）
    体验完整的智能助手 Agent
""")
