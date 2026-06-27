"""
第 04 章 消息与提示词模板 练习 1：消息类型与多轮对话
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 四种消息类型：System / Human / AI / Tool
  2. 消息格式：JSON 字典格式 vs 对象格式
  3. 消息对象字段说明（content, name 等）
  4. 对话历史管理（关键规则：每次调用必须传递完整历史）
  5. 对话历史优化（保留最近 N 条消息）
  6. 多轮对话聊天机器人（模拟）

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/llm_basics/05_消息类型与多轮对话.py"
"""

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


# =========================================================
# 1. 四种消息类型
# =========================================================

print("=" * 60)
print("练习 1：四种消息类型")
print("=" * 60)

# 1. SystemMessage：设定 AI 的角色和行为
sys_msg = SystemMessage(content="你是一个 Python 编程导师，回答简洁准确。")
print(f"SystemMessage: {sys_msg}")
print(f"  type: {sys_msg.type}")       # 消息类型标识
print(f"  type: {sys_msg.type}")       # 消息类型
print(f"  content: {sys_msg.content}")

# 2. HumanMessage：用户的输入
human_msg = HumanMessage(content="什么是列表推导式？")
print(f"\nHumanMessage: {human_msg}")
print(f"  type: {human_msg.type}")
print(f"  content: {human_msg.content}")

# 3. AIMessage：模型的回复
ai_msg = AIMessage(content="列表推导式是一种简洁创建列表的方式，如 [x**2 for x in range(10)]")
print(f"\nAIMessage: {ai_msg}")
print(f"  type: {ai_msg.type}")
print(f"  content: {ai_msg.content}")

# 4. ToolMessage：工具调用的返回结果
tool_msg = ToolMessage(
    content="北京晴天，温度 15°C",
    tool_call_id="call_001",  # 必须提供，关联对应的工具调用
)
print(f"\nToolMessage: {tool_msg}")
print(f"  type: {tool_msg.type}")
print(f"  content: {tool_msg.content}")
print(f"  tool_call_id: {tool_msg.tool_call_id}")


# =========================================================
# 2. 消息格式：JSON 字典格式 vs 对象格式
# =========================================================

print("\n" + "=" * 60)
print("练习 2：消息格式对比（JSON 字典 vs 对象）")
print("=" * 60)

# 格式 1：JSON 字典格式（适合从 API 直接构造）
messages_json = [
    {"role": "system", "content": "你是一个翻译助手。"},
    {"role": "user", "content": "把'你好'翻译成英文。"},
    {"role": "assistant", "content": "Hello!"},
]
print("JSON 字典格式:")
for m in messages_json:
    print(f"  {m}")

# 格式 2：对象格式（LangChain 推荐，类型安全）
messages_obj = [
    SystemMessage(content="你是一个翻译助手。"),
    HumanMessage(content="把'你好'翻译成英文。"),
    AIMessage(content="Hello!"),
]
print(f"\n对象格式:")
for m in messages_obj:
    print(f"  {type(m).__name__}(content='{m.content}')")

# 两种格式可以互相转换
print(f"\n对象转字典: {messages_obj[1].model_dump()}")


# =========================================================
# 3. 消息对象的 name 字段（多人对话场景）
# =========================================================

print("\n" + "=" * 60)
print("练习 3：name 字段（多人对话场景）")
print("=" * 60)

# name 字段可以标识消息的发送者，在多人对话场景中很有用
messages_with_name = [
    SystemMessage(content="请列出谁说了什么，不要判断对错。"),
    HumanMessage(content="我认为 1+1=2", name="Bob"),
    HumanMessage(content="我认为 1+1>2", name="Tom"),
    HumanMessage(content="请总结以上观点", name="audience"),
]

print("多人对话消息列表:")
for m in messages_with_name:
    name = getattr(m, "name", None) or "system"
    print(f"  [{name}] {m.content}")


# =========================================================
# 4. 对话历史管理（核心规则）
# =========================================================

print("\n" + "=" * 60)
print("练习 4：对话历史管理")
print("=" * 60)

# 关键规则：每次调用模型必须传递完整的对话历史！
# 因为大模型没有记忆，它的输出只和输入的内容（上下文）有关

# 模拟一个对话历史
chat_history = [
    SystemMessage(content="你是一个友好的助手。"),
]

# 第 1 轮对话
user_input_1 = "你好！我叫张三。"
chat_history.append(HumanMessage(content=user_input_1))
# 模拟模型回复（实际开发中用 model.invoke(chat_history)）
ai_reply_1 = AIMessage(content="你好张三！很高兴认识你！有什么可以帮你的吗？")
chat_history.append(ai_reply_1)

print(f"第 1 轮:")
print(f"  👤 用户: {user_input_1}")
print(f"  🤖 AI: {ai_reply_1.content}")

# 第 2 轮对话（模型需要看到之前的完整历史才能"记住"用户的名字）
user_input_2 = "我刚才告诉你我叫什么？"
chat_history.append(HumanMessage(content=user_input_2))
ai_reply_2 = AIMessage(content="你刚才告诉我你叫张三。")

print(f"\n第 2 轮:")
print(f"  👤 用户: {user_input_2}")
print(f"  🤖 AI: {ai_reply_2.content}")
chat_history.append(ai_reply_2)

# 第 3 轮对话
user_input_3 = "用我的名字跟我打招呼。"
chat_history.append(HumanMessage(content=user_input_3))
ai_reply_3 = AIMessage(content="张三你好！很高兴再次和你交流！")

print(f"\n第 3 轮:")
print(f"  👤 用户: {user_input_3}")
print(f"  🤖 AI: {ai_reply_3.content}")
chat_history.append(ai_reply_3)

print(f"\n完整对话历史（{len(chat_history)} 条消息）:")
for i, m in enumerate(chat_history):
    role = type(m).__name__.replace("Message", "")
    print(f"  [{i}] {role}: {m.content}")


# =========================================================
# 5. 对话历史优化：保留最近 N 条消息
# =========================================================

print("\n" + "=" * 60)
print("练习 5：对话历史优化（保留最近 N 条）")
print("=" * 60)

# 问题：对话历史会越来越长，消耗大量 tokens 和成本
# 解决方案：只保留最近的消息 + 系统消息


def keep_recent_messages(history, max_messages=6):
    """
    保留系统消息 + 最近 N 条对话消息
    """
    # 分离系统消息和对话消息
    system_msgs = [m for m in history if isinstance(m, SystemMessage)]
    chat_msgs = [m for m in history if not isinstance(m, SystemMessage)]

    # 保留最近的对话消息（确保是偶数对：user + ai）
    recent_chat = chat_msgs[-max_messages:]

    return system_msgs + recent_chat


# 构造一个很长的对话历史
long_history = [SystemMessage(content="你是助手。")]
for i in range(10):  # 10 轮对话 = 20 条消息
    long_history.append(HumanMessage(content=f"这是第 {i+1} 个问题。"))
    long_history.append(AIMessage(content=f"这是第 {i+1} 个回答。"))

print(f"优化前的消息数: {len(long_history)}")

optimized = keep_recent_messages(long_history, max_messages=6)
print(f"优化后的消息数: {len(optimized)}")
print(f"\n优化后保留的消息:")
for m in optimized:
    role = type(m).__name__.replace("Message", "")
    print(f"  [{role}] {m.content}")


# =========================================================
# 6. 多轮对话聊天机器人（完整模拟）
# =========================================================

print("\n" + "=" * 60)
print("练习 6：多轮对话聊天机器人（模拟）")
print("=" * 60)


class SimpleChatBot:
    """简单的多轮对话聊天机器人（模拟版，不调用真实 API）"""

    def __init__(self, system_prompt: str, max_history: int = 6):
        self.system_msg = SystemMessage(content=system_prompt)
        self.history = [self.system_msg]
        self.max_history = max_history

    def chat(self, user_input: str) -> str:
        """进行一轮对话"""
        # 添加用户消息
        self.history.append(HumanMessage(content=user_input))

        # 模拟 AI 回复（实际开发中：response = model.invoke(self.history)）
        ai_reply = self._mock_reply(user_input)

        # 添加 AI 回复
        self.history.append(AIMessage(content=ai_reply))

        # 优化历史
        self._optimize_history()

        return ai_reply

    def _mock_reply(self, user_input: str) -> str:
        """模拟 AI 回复（实际开发中替换为 model.invoke）"""
        if "名字" in user_input or "叫什么" in user_input:
            # 从历史中查找用户之前提到的名字
            for m in self.history:
                if isinstance(m, HumanMessage) and "叫" in m.content:
                    return f"根据之前的对话，你告诉我你叫{m.content.split('叫')[-1].rstrip('。.!！')}"
            return "你还没有告诉我你的名字呢！"
        elif "再见" in user_input:
            return "再见！期待下次和你聊天！"
        else:
            return f"我收到了你的消息：'{user_input}'。这是一个模拟回复。"

    def _optimize_history(self):
        """保留系统消息 + 最近 N 条对话"""
        chat_msgs = [m for m in self.history if not isinstance(m, SystemMessage)]
        recent = chat_msgs[-self.max_history:]
        self.history = [self.system_msg] + recent

    def show_history(self):
        """显示当前对话历史"""
        print(f"\n当前对话历史（{len(self.history)} 条消息）:")
        for m in self.history:
            role = type(m).__name__.replace("Message", "")
            print(f"  [{role}] {m.content[:50]}{'...' if len(m.content) > 50 else ''}")


# 使用聊天机器人
bot = SimpleChatBot(system_prompt="你是一个友好的聊天助手。", max_history=6)

print("第 1 轮:")
reply = bot.chat("你好！我叫张三。")
print(f"  👤 用户: 你好！我叫张三。")
print(f"  🤖 AI: {reply}")

print("\n第 2 轮:")
reply = bot.chat("你还记得我叫什么吗？")
print(f"  👤 用户: 你还记得我叫什么吗？")
print(f"  🤖 AI: {reply}")

print("\n第 3 轮:")
reply = bot.chat("再说一次我的名字。")
print(f"  👤 用户: 再说一次我的名字。")
print(f"  🤖 AI: {reply}")

bot.show_history()


# =========================================================
# 7. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 04 章消息部分的核心知识点：
  1. 四种消息类型：System / Human / AI / Tool
  2. 两种格式：JSON 字典格式 vs 对象格式
  3. name 字段：多人对话场景标识发送者
  4. 对话历史管理：每次调用必须传递完整历史！
  5. 对话历史优化：保留系统消息 + 最近 N 条
  6. 多轮对话机器人：历史拼接 + 历史优化

关键规则：
  大模型没有记忆！它的输出只和输入的上下文有关。
  所以每次调用都必须传递完整的对话历史。

下一步：运行 06_提示词模板.py，学习 PromptTemplate！
""")
