"""
第 07 章 智能体 练习 3：综合实战 - 智能助手 Agent
=================================================
本练习需要大模型 API，是一个完整的 Agent 实战项目。
学习目标：
  1. 构建多功能智能助手（天气 + 计算 + 货币 + 时间）
  2. Agent 自主选择工具完成任务
  3. 交互式多轮对话
  4. 流式输出实时查看执行过程

前置条件：
  1. 将 configs/.env.example 复制为 configs/.env
  2. 填入你的 API Key（推荐 DeepSeek）

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/agents/03_综合实战_智能助手.py"
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


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
# 2. 定义工具集
# =========================================================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息

    Args:
        city: 城市名称，如 "北京"、"上海"、"广州"

    Returns:
        天气信息字符串
    """
    weather_data = {
        "北京": "晴天，温度 15°C，湿度 40%，西北风 3级",
        "上海": "多云，温度 22°C，湿度 65%，东南风 2级",
        "广州": "小雨，温度 28°C，湿度 80%，南风 3级",
        "深圳": "阴天，温度 26°C，湿度 70%，东风 2级",
        "杭州": "晴转多云，温度 20°C，湿度 55%，微风",
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式，支持加减乘除、三角函数、对数、平方根等

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"、"sqrt(144)"、"sin(pi/2)"

    Returns:
        计算结果字符串
    """
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
        amount: 金额数量
        from_curr: 源货币代码（CNY/USD/EUR/GBP/JPY/HKD）
        to_curr: 目标货币代码（CNY/USD/EUR/GBP/JPY/HKD）

    Returns:
        转换结果字符串
    """
    rates = {"CNY": 1, "USD": 7.2, "EUR": 7.8, "GBP": 9.1, "JPY": 0.048, "HKD": 0.92}
    from_rate = rates.get(from_curr)
    to_rate = rates.get(to_curr)
    if not from_rate or not to_rate:
        return f"不支持的货币代码，支持：{list(rates.keys())}"
    result = amount * from_rate / to_rate
    return f"{amount} {from_curr} = {result:.2f} {to_curr}"


@tool
def get_time_info() -> str:
    """
    获取当前时间信息，包括日期、时间、星期

    Returns:
        当前时间信息字符串
    """
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return (
        f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"{weekdays[now.weekday()]}\n"
        f"今年第 {now.timetuple().tm_yday} 天"
    )


tools = [get_weather, calculate, convert_currency, get_time_info]
print(f"✅ 已准备 {len(tools)} 个工具: 天气查询、数学计算、货币转换、时间查询")


# =========================================================
# 3. 创建智能助手 Agent
# =========================================================

agent = create_agent(
    model=model,
    tools=tools,
    prompt=(
        "你是一个智能助手，可以帮助用户查询天气、进行数学计算、"
        "货币转换和查询时间。请根据用户的问题，自动选择合适的工具。"
        "回答时用简洁的中文，并在使用工具后给出清晰的总结。"
    ),
    name="smart_assistant",
)
print("✅ 智能助手 Agent 创建成功\n")


# =========================================================
# 4. 自动化测试场景
# =========================================================

print("=" * 60)
print("场景测试：Agent 自动选择工具")
print("=" * 60)

test_cases = [
    "现在几点了？",
    "北京天气怎么样？",
    "帮我算一下 sqrt(144) + sin(pi/2)",
    "100 美元等于多少人民币？",
    "查一下北京和上海的天气，然后算一下两地温度差",
]

for i, question in enumerate(test_cases, 1):
    print(f"\n{'─' * 50}")
    print(f"场景 {i}: {question}")
    print(f"{'─' * 50}")

    # 流式输出，实时查看工具调用过程
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    msg_type = type(msg).__name__
                    if msg_type == "AIMessage":
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  🔧 调用工具: {tc['name']}({tc['args']})")
                        else:
                            print(f"  🤖 回复: {msg.content}")
                    elif msg_type == "ToolMessage":
                        print(f"  📦 结果: {msg.content}")


# =========================================================
# 5. 交互式多轮对话
# =========================================================

print("\n" + "=" * 60)
print("交互式对话（输入 'quit' 退出）")
print("=" * 60)

# 保留对话历史
chat_history = []

while True:
    try:
        user_input = input("\n👤 你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q", "退出"):
        print("再见！")
        break

    # 添加用户消息到历史
    chat_history.append(HumanMessage(content=user_input))

    # 调用 Agent（传入完整历史）
    print("🤖 助手: ", end="", flush=True)

    final_response = None
    for chunk in agent.stream(
        {"messages": chat_history},
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    msg_type = type(msg).__name__
                    if msg_type == "AIMessage":
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"\n  🔧 调用工具: {tc['name']}({tc['args']})")
                        else:
                            print(f"{msg.content}")
                            final_response = msg
                    elif msg_type == "ToolMessage":
                        print(f"  📦 结果: {msg.content}")

    # 更新对话历史（保留最近 10 条）
    if final_response:
        chat_history.append(final_response)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


# =========================================================
# 6. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
综合实战完成！你已构建了一个多功能智能助手 Agent：

  功能：
    🌤  天气查询  — 查询城市天气
    🔢  数学计算  — 四则运算、三角函数、对数等
    💱  货币转换  — 多国货币汇率转换
    ⏰  时间查询  — 当前日期时间

  Agent 能力：
    ✅ 自动选择合适的工具
    ✅ 多轮工具调用（一个任务中调用多个工具）
    ✅ 流式输出（实时查看执行过程）
    ✅ 多轮对话（保留上下文）

  关键经验：
    1. 工具数量 2-5 个最佳，太多会混淆
    2. 工具 docstring 要清晰，Agent 通过它理解用途
    3. 系统提示词定义 Agent 的行为规范
    4. 流式输出提升用户体验
    5. 对话历史需要管理（保留最近 N 条）

下一步学习建议：
    - 第 08 章：中间件（动态工具绑定、请求拦截）
    - 第 09 章：上下文与记忆（持久化对话历史）
    - 第 10 章：RAG（检索增强生成）
""")
