"""
第 09 章 上下文与记忆 练习：记忆机制与对话历史管理
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 理解为什么需要记忆（Memory）
  2. 短期记忆 vs 长期记忆
  3. 对话历史管理策略
  4. 检查点（Checkpoint）机制
  5. 模拟持久化存储
  6. Memory 实现方式对比

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/agents/05_上下文与记忆.py"
"""

import json
import hashlib
from datetime import datetime
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)


# =========================================================
# 1. 为什么需要记忆
# =========================================================

print("=" * 60)
print("练习 1：为什么需要记忆（Memory）")
print("=" * 60)

print("""
问题：大模型没有记忆！

  大模型的输出只和输入的内容（上下文）有关。
  每次调用都是独立的，模型不会"记住"之前的对话。

  第 1 次调用：用户说"我叫张三" → 模型回复"你好张三"
  第 2 次调用：用户说"我叫什么？" → 模型回复"我不知道你的名字"
                                    ↑ 因为第 2 次调用没有第 1 次的上下文

解决方案：每次调用都把历史消息传给模型

  对话历史 → 作为 messages 传入 → 模型"看到"之前的对话 → "记住"了

但这带来新问题：
  ❶ 对话越来越长 → token 消耗和成本激增
  ❷ 程序重启后历史丢失 → 需要持久化存储
  ❸ 多用户场景 → 需要区分不同用户的会话

这就是 Memory（记忆）要解决的问题！
""")


# =========================================================
# 2. 短期记忆 vs 长期记忆
# =========================================================

print("=" * 60)
print("练习 2：短期记忆 vs 长期记忆")
print("=" * 60)

memory_types = {
    "短期记忆（Short-term Memory）": {
        "存储位置": "内存（RAM）",
        "生命周期": "程序运行期间，重启后丢失",
        "实现方式": "消息列表（messages list）",
        "适用场景": "单次会话、临时对话",
        "LangChain": "Agent 的 state['messages']",
    },
    "长期记忆（Long-term Memory）": {
        "存储位置": "外部存储（数据库/文件）",
        "生命周期": "持久化，跨会话保留",
        "实现方式": "检查点（Checkpoint）+ 持久化后端",
        "适用场景": "多会话、用户偏好、知识积累",
        "LangChain": "Checkpointer + PostgreSQL/SQLite",
    },
}

for mem_type, info in memory_types.items():
    print(f"\n{mem_type}")
    for key, value in info.items():
        print(f"  {key}: {value}")


# =========================================================
# 3. 对话历史管理策略
# =========================================================

print("\n" + "=" * 60)
print("练习 3：对话历史管理策略")
print("=" * 60)


class ConversationMemory:
    """对话记忆管理器（模拟短期记忆）"""

    def __init__(self, system_prompt: str, max_messages: int = 10):
        self.system_msg = SystemMessage(content=system_prompt)
        self.history = []
        self.max_messages = max_messages

    def add_user_message(self, content):
        self.history.append(HumanMessage(content=content))

    def add_ai_message(self, content):
        self.history.append(AIMessage(content=content))

    def get_messages(self):
        """获取完整消息列表（系统消息 + 历史）"""
        return [self.system_msg] + self.history

    def optimize(self):
        """策略 1：保留最近 N 条消息"""
        self.history = self.history[-self.max_messages:]

    def summarize(self):
        """策略 2：总结旧消息（模拟）"""
        if len(self.history) <= self.max_messages:
            return
        # 把旧消息总结为一条
        old_msgs = self.history[:-self.max_messages]
        summary = f"[之前对话总结：共{len(old_msgs)}条消息，讨论了相关话题]"
        self.history = [SystemMessage(content=summary)] + self.history[-self.max_messages:]


# 演示策略 1：保留最近 N 条
print("--- 策略 1：保留最近 N 条消息 ---")
mem = ConversationMemory("你是助手", max_messages=4)
for i in range(6):
    mem.add_user_message(f"第 {i+1} 个问题")
    mem.add_ai_message(f"第 {i+1} 个回答")

print(f"优化前: {len(mem.history)} 条消息")
mem.optimize()
print(f"优化后: {len(mem.history)} 条消息")
for m in mem.history:
    print(f"  {type(m).__name__}: {m.content}")

# 演示策略 2：总结旧消息
print("\n--- 策略 2：总结旧消息 ---")
mem2 = ConversationMemory("你是助手", max_messages=4)
for i in range(6):
    mem2.add_user_message(f"第 {i+1} 个问题")
    mem2.add_ai_message(f"第 {i+1} 个回答")

print(f"总结前: {len(mem2.history)} 条消息")
mem2.summarize()
print(f"总结后: {len(mem2.history)} 条消息")
for m in mem2.history:
    print(f"  {type(m).__name__}: {m.content}")


# =========================================================
# 4. 检查点（Checkpoint）机制模拟
# =========================================================

print("\n" + "=" * 60)
print("练习 4：检查点（Checkpoint）机制模拟")
print("=" * 60)


class Checkpoint:
    """检查点：保存某一时刻的完整状态"""

    def __init__(self, checkpoint_id, messages, metadata=None):
        self.checkpoint_id = checkpoint_id
        self.messages = messages
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "checkpoint_id": self.checkpoint_id,
            "messages": [str(m) for m in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class MemorySaver:
    """模拟 LangChain 的 MemorySaver（内存检查点存储）"""

    def __init__(self):
        self.checkpoints = {}

    def save(self, thread_id, checkpoint):
        """保存检查点"""
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = []
        self.checkpoints[thread_id].append(checkpoint)
        print(f"  💾 保存检查点: thread={thread_id}, id={checkpoint.checkpoint_id}")

    def load(self, thread_id):
        """加载最新的检查点"""
        if thread_id not in self.checkpoints or not self.checkpoints[thread_id]:
            return None
        cp = self.checkpoints[thread_id][-1]
        print(f"  📂 加载检查点: thread={thread_id}, id={cp.checkpoint_id}")
        return cp

    def list_checkpoints(self, thread_id):
        """列出所有检查点"""
        return self.checkpoints.get(thread_id, [])


# 模拟带检查点的对话
saver = MemorySaver()

# 线程 1：用户 A 的对话
thread_id = "user_A_session_1"
history_a = [SystemMessage(content="你是助手")]

# 第 1 轮
history_a.append(HumanMessage(content="我叫张三"))
history_a.append(AIMessage(content="你好张三！"))
cp1 = Checkpoint("cp_001", history_a.copy(), {"step": 1})
saver.save(thread_id, cp1)

# 第 2 轮
history_a.append(HumanMessage(content="我在学LangChain"))
history_a.append(AIMessage(content="太好了！LangChain是一个强大的框架。"))
cp2 = Checkpoint("cp_002", history_a.copy(), {"step": 2})
saver.save(thread_id, cp2)

# 模拟程序重启 → 从检查点恢复
print("\n--- 模拟程序重启，从检查点恢复 ---")
restored = saver.load(thread_id)
print(f"  恢复的消息数: {len(restored.messages)}")
for m in restored.messages:
    print(f"  {type(m).__name__}: {m.content}")

# 线程 2：用户 B 的对话（独立于用户 A）
thread_id_b = "user_B_session_1"
history_b = [SystemMessage(content="你是助手"), HumanMessage(content="你好"), AIMessage(content="你好！")]
saver.save(thread_id_b, Checkpoint("cp_001", history_b, {"step": 1}))


# =========================================================
# 5. 持久化存储模拟（SQLite 风格）
# =========================================================

print("\n" + "=" * 60)
print("练习 5：持久化存储模拟")
print("=" * 60)


class PersistentCheckpointer:
    """模拟持久化检查点存储（类似 SQLite/PostgreSQL）"""

    def __init__(self):
        self.storage = {}  # 模拟数据库表

    def _make_id(self, thread_id, messages):
        """生成唯一 ID"""
        content = thread_id + str(len(messages)) + str(messages[-1].content[:20])
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def put(self, thread_id, messages, metadata=None):
        """写入检查点"""
        cp_id = self._make_id(thread_id, messages)
        record = {
            "thread_id": thread_id,
            "checkpoint_id": cp_id,
            "messages": [{"role": type(m).__name__, "content": m.content} for m in messages],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.storage[cp_id] = record
        return cp_id

    def get(self, thread_id):
        """获取线程的最新检查点"""
        latest = None
        for record in self.storage.values():
            if record["thread_id"] == thread_id:
                if latest is None or record["timestamp"] > latest["timestamp"]:
                    latest = record
        return latest

    def list_by_thread(self, thread_id):
        """列出线程的所有检查点"""
        return [r for r in self.storage.values() if r["thread_id"] == thread_id]


# 使用持久化存储
db = PersistentCheckpointer()

# 保存对话
msgs = [
    SystemMessage(content="你是Python导师"),
    HumanMessage(content="什么是装饰器？"),
    AIMessage(content="装饰器是一种用@语法的函数包装器..."),
]
cp_id = db.put("session_001", msgs, {"step": 1, "topic": "decorator"})
print(f"写入检查点: {cp_id}")

# 模拟"重启"后读取
print("\n--- 模拟重启后读取 ---")
restored = db.get("session_001")
print(f"检查点 ID: {restored['checkpoint_id']}")
print(f"时间戳: {restored['timestamp']}")
print(f"消息数: {len(restored['messages'])}")
for m in restored["messages"]:
    print(f"  {m['role']}: {m['content']}")


# =========================================================
# 6. LangChain Memory 使用方式对比
# =========================================================

print("\n" + "=" * 60)
print("练习 6：LangChain Memory 使用方式对比")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│  方式              │  存储后端        │  适用场景            │
├─────────────────────────────────────────────────────────────┤
│  MemorySaver       │  内存            │  开发调试、单次运行   │
│  SQLiteSaver       │  SQLite 文件     │  单机持久化、轻量级   │
│  PostgresSaver     │  PostgreSQL      │  生产环境、多用户     │
│  RedisSaver        │  Redis           │  高性能、TTL 过期     │
└─────────────────────────────────────────────────────────────┘

# 使用 MemorySaver（内存，开发调试用）
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)

# 调用时指定 thread_id（区分不同用户/会话）
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config={"configurable": {"thread_id": "user_001"}},
)

# 下次调用会自动恢复历史
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config={"configurable": {"thread_id": "user_001"}},
)
# → Agent 能回答"你叫张三"，因为检查点保存了历史

# 使用 SQLiteSaver（持久化到文件）
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 使用 PostgresSaver（生产环境）
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/dbname"
)
""")


# =========================================================
# 7. 练习小结
# =========================================================

print("=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 09 章的核心知识点：

  1. 记忆的本质：每次调用传递完整历史（大模型本身没有记忆）
  2. 短期记忆：内存中的消息列表，重启后丢失
  3. 长期记忆：检查点 + 持久化存储，跨会话保留
  4. 历史管理策略：
     - 保留最近 N 条（简单截断）
     - 总结旧消息（SummarizationMiddleware）
  5. 检查点机制：
     - 保存某一时刻的完整状态
     - 通过 thread_id 区分不同会话
     - 重启后可从检查点恢复
  6. 持久化后端：MemorySaver / SQLiteSaver / PostgresSaver

关键概念：
  - thread_id：标识一个独立的对话会话
  - checkpoint：保存对话状态的快照
  - checkpointer：管理检查点的存储和读取
""")
