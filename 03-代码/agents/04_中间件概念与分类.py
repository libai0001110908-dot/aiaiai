"""
第 08 章 中间件 练习：中间件概念与分类
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 理解中间件的作用和执行位置
  2. LangChain 内置中间件的六大分类
  3. 自定义中间件的结构
  4. 模拟中间件执行流程

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/agents/04_中间件概念与分类.py"
"""

import time
from langchain_core.messages import HumanMessage, AIMessage


# =========================================================
# 1. 中间件概述
# =========================================================

print("=" * 60)
print("练习 1：中间件概述")
print("=" * 60)

print("""
中间件（Middleware）是什么？

  在 Agent 执行循环中设置"钩子（hooks）"，让你在不改 Agent
  主体逻辑的情况下实现策略与治理。

  类比：中间件就像快递流水线上的"质检站"，
        不改变快递本身，但在关键节点做检查和处理。

  没有 Middleware 的 Agent：
    用户 → Agent(模型+工具) → 用户

  有 Middleware 的 Agent：
    用户 → [中间件1] → [中间件2] → Agent(模型+工具) → [中间件3] → 用户

中间件的执行位置（钩子点）：
  ① 模型调用前（before_model）
  ② 模型调用后（after_model）
  ③ 工具调用前（before_tool）
  ④ 工具调用后（after_tool）

核心价值：
  ✅ 关注点分离 — 业务逻辑与治理逻辑解耦
  ✅ 可组合    — 多个中间件可以叠加使用
  ✅ 可复用    — 同一中间件可用于不同 Agent
""")


# =========================================================
# 2. 内置中间件六大分类
# =========================================================

print("=" * 60)
print("练习 2：内置中间件六大分类")
print("=" * 60)

middleware_categories = {
    "1. 上下文管理类": {
        "中间件": ["SummarizationMiddleware", "ContextEditingMiddleware"],
        "作用": "自动总结历史、裁剪上下文，减少 token 消耗",
        "触发条件": "tokens（累计token数）/ messages（消息条数）",
    },
    "2. 安全治理类": {
        "中间件": ["PIIMiddleware", "ToolApprovalMiddleware"],
        "作用": "敏感信息脱敏、工具调用审批（人工确认）",
        "触发条件": "工具调用前拦截，需人工审批",
    },
    "3. 任务管理类": {
        "中间件": ["TodoListMiddleware"],
        "作用": "自动生成任务清单，按步骤执行",
        "类比": "从'想到哪写到哪'变为'先列CheckList再执行'",
    },
    "4. 限制控制类": {
        "中间件": ["RateLimitMiddleware", "ToolCallLimitMiddleware"],
        "作用": "限制调用频率、限制工具调用次数",
        "参数": "run_limit（单次运行限制）/ session_limit（会话限制）",
    },
    "5. 执行能力扩展类": {
        "中间件": ["自定义中间件"],
        "作用": "给 Agent 更多'手脚'，如日志记录、性能监控",
    },
    "6. 模型供应商相关类": {
        "中间件": ["各供应商特定中间件"],
        "作用": "特定模型的适配与优化",
    },
}

for category, info in middleware_categories.items():
    print(f"\n{category}")
    print(f"  中间件: {info['中间件']}")
    print(f"  作用: {info['作用']}")
    if "触发条件" in info:
        print(f"  触发: {info['触发条件']}")
    if "参数" in info:
        print(f"  参数: {info['参数']}")
    if "类比" in info:
        print(f"  类比: {info['类比']}")


# =========================================================
# 3. 模拟中间件执行流程
# =========================================================

print("\n" + "=" * 60)
print("练习 3：模拟中间件执行流程")
print("=" * 60)


class MockMiddleware:
    """模拟中间件的基类"""

    def __init__(self, name):
        self.name = name

    def before_model(self, state):
        """模型调用前"""
        print(f"  [{self.name}] before_model: 检查输入...")
        return state

    def after_model(self, state, response):
        """模型调用后"""
        print(f"  [{self.name}] after_model: 处理模型输出...")
        return response

    def before_tool(self, tool_name, tool_args):
        """工具调用前"""
        print(f"  [{self.name}] before_tool({tool_name}): 审批工具调用...")
        return True  # True=允许, False=中断

    def after_tool(self, tool_name, result):
        """工具调用后"""
        print(f"  [{self.name}] after_tool({tool_name}): 记录工具结果...")
        return result


class LoggingMiddleware(MockMiddleware):
    """日志记录中间件"""

    def before_model(self, state):
        print(f"  📝 [LoggingMiddleware] 记录请求: {state.get('messages', [{}])[-1]}")
        return state

    def after_model(self, state, response):
        print(f"  📝 [LoggingMiddleware] 记录响应: {str(response)[:50]}...")
        return response


class TimingMiddleware(MockMiddleware):
    """性能计时中间件"""

    def before_model(self, state):
        self.start_time = time.time()
        print(f"  ⏱  [TimingMiddleware] 开始计时...")
        return state

    def after_model(self, state, response):
        elapsed = time.time() - self.start_time
        print(f"  ⏱  [TimingMiddleware] 耗时: {elapsed:.4f}秒")
        return response


class ToolApprovalMiddleware(MockMiddleware):
    """工具审批中间件（模拟人工确认）"""

    def __init__(self, name, auto_approve=True):
        super().__init__(name)
        self.auto_approve = auto_approve

    def before_tool(self, tool_name, tool_args):
        if self.auto_approve:
            print(f"  ✅ [ToolApprovalMiddleware] 自动批准 {tool_name}({tool_args})")
            return True
        else:
            print(f"  ⚠️  [ToolApprovalMiddleware] 需要人工审批 {tool_name}({tool_args})")
            return False  # 中断执行


# 模拟 Agent 执行流程
def mock_agent_execute(user_input, middlewares):
    """模拟 Agent 执行（带中间件）"""
    state = {"messages": [HumanMessage(content=user_input)]}

    print(f"\n👤 用户: {user_input}")

    # before_model 钩子
    print("📌 before_model 钩子:")
    for mw in middlewares:
        state = mw.before_model(state)

    # 模拟模型调用
    print("  🤖 [模型调用中...]")
    time.sleep(0.1)  # 模拟推理时间
    response = AIMessage(content=f"这是对'{user_input}'的回复")

    # after_model 钩子
    print("📌 after_model 钩子:")
    for mw in middlewares:
        response = mw.after_model(state, response)

    print(f"🤖 Agent: {response.content}")
    return response


# 使用中间件
middlewares = [
    LoggingMiddleware("logging"),
    TimingMiddleware("timing"),
    ToolApprovalMiddleware("approval", auto_approve=True),
]

mock_agent_execute("北京天气怎么样？", middlewares)
mock_agent_execute("帮我算一下 2+3", middlewares)


# =========================================================
# 4. 中间件叠加顺序演示
# =========================================================

print("\n" + "=" * 60)
print("练习 4：中间件叠加顺序")
print("=" * 60)

print("""
中间件的执行顺序（洋葱模型）：

  请求 →  [MW1.before] → [MW2.before] → [MW3.before] → Agent
  响应 ←  [MW1.after]  ← [MW2.after]  ← [MW3.after]  ← Agent

  before 钩子：按添加顺序执行（外→内）
  after 钩子：按添加逆序执行（内→外）

示例：添加顺序为 [Logging, Timing, Approval]

  请求 → Logging.before → Timing.before → Approval.before → Agent
  响应 ← Logging.after  ← Timing.after  ← Approval.after  ← Agent
""")

# 演示不同顺序的效果
print("--- 顺序 A: [Logging, Timing] ---")
mock_agent_execute("测试A", [LoggingMiddleware("log"), TimingMiddleware("time")])

print("\n--- 顺序 B: [Timing, Logging] ---")
mock_agent_execute("测试B", [TimingMiddleware("time"), LoggingMiddleware("log")])


# =========================================================
# 5. 自定义中间件结构
# =========================================================

print("\n" + "=" * 60)
print("练习 5：自定义中间件结构")
print("=" * 60)

print("""
# LangChain 中自定义中间件的典型结构：

from langchain.agents.middleware import AgentMiddleware

class MyCustomMiddleware(AgentMiddleware):
    \"\"\"自定义中间件\"\"\"

    def before_model(self, state):
        # 模型调用前的逻辑
        # 例如：日志记录、输入校验、上下文裁剪
        return state

    def after_model(self, state, response):
        # 模型调用后的逻辑
        # 例如：输出过滤、结果转换
        return response

    def before_tool(self, tool_call):
        # 工具调用前的逻辑
        # 例如：权限检查、参数校验、人工审批
        return True  # True=允许, False=中断

    def after_tool(self, tool_call, result):
        # 工具调用后的逻辑
        # 例如：结果记录、错误处理
        return result


# 使用中间件
agent = create_agent(
    model=model,
    tools=tools,
    prompt="你是助手",
    middleware=[
        MyCustomMiddleware(),
        SummarizationMiddleware(max_tokens=1000),
        ToolApprovalMiddleware(auto_approve=False),
    ],
)
""")


# =========================================================
# 6. 常用中间件使用示例（伪代码）
# =========================================================

print("=" * 60)
print("练习 6：常用中间件使用示例（伪代码）")
print("=" * 60)

print("""
# 1. SummarizationMiddleware — 自动总结历史
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        SummarizationMiddleware(
            max_tokens=1000,        # token 数触发阈值
            # max_messages=20,      # 或按消息条数触发
        )
    ],
)

# 2. ToolApprovalMiddleware — 工具调用审批
from langchain.agents.middleware import ToolApprovalMiddleware

agent = create_agent(
    model=model,
    tools=[send_email, delete_file],
    middleware=[
        ToolApprovalMiddleware(
            auto_approve=False,     # 需要人工审批
        )
    ],
)

# 3. ToolCallLimitMiddleware — 限制工具调用次数
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        ToolCallLimitMiddleware(
            run_limit=3,            # 每次运行最多 3 次工具调用
            session_limit=10,       # 整个会话最多 10 次
        )
    ],
)

# 4. PIIMiddleware — 敏感信息脱敏
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        PIIMiddleware(
            apply_to_input=True,    # 输入时检测
            apply_to_output=True,   # 输出时检测
        )
    ],
)
""")


# =========================================================
# 7. 练习小结
# =========================================================

print("=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 08 章的核心知识点：

  1. 中间件概念：在 Agent 执行循环中设置钩子，不改主体逻辑
  2. 四个钩子点：before_model / after_model / before_tool / after_tool
  3. 六大分类：
     - 上下文管理（Summarization / ContextEditing）
     - 安全治理（PII / ToolApproval）
     - 任务管理（TodoList）
     - 限制控制（RateLimit / ToolCallLimit）
     - 执行能力扩展（自定义）
     - 模型供应商相关
  4. 洋葱模型：before 顺序执行，after 逆序执行
  5. 自定义中间件结构

最佳实践：
  - 只添加必要的中间件，过多会影响性能
  - 安全相关中间件放在最外层
  - 日志/计时中间件放在最内层
""")
