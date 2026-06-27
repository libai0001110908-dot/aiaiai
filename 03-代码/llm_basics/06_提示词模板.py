"""
第 04 章 消息与提示词模板 练习 2：提示词模板
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. PromptTemplate：基础字符串模板
  2. ChatPromptTemplate：聊天消息模板（推荐）
  3. from_messages()：推荐的创建方式
  4. 三种调用方式：invoke() / format() / format_messages()
  5. partial()：部分变量预填充
  6. MessagesPlaceholder：消息占位符
  7. 可复用模板库与模板组合

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/llm_basics/06_提示词模板.py"
"""

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


# =========================================================
# 1. PromptTemplate：基础字符串模板
# =========================================================

print("=" * 60)
print("练习 1：PromptTemplate 基础模板")
print("=" * 60)

# 创建模板（变量用 {变量名} 占位）
prompt = PromptTemplate.from_template(
    "请用 {tone} 的语气，解释 {concept} 这个概念。"
)

# 方式 1：format() → 返回字符串
result1 = prompt.format(tone="通俗易懂", concept="机器学习")
print(f"format() 结果: {result1}")

# 方式 2：invoke() → 返回 StringPromptValue
result2 = prompt.invoke({"tone": "专业严谨", "concept": "深度学习"})
print(f"invoke() 结果: {result2}")

# PromptTemplate 适合生成纯文本，但不支持角色区分
print(f"\n输出格式: 纯文本字符串（不支持 system/user/assistant 角色）")


# =========================================================
# 2. ChatPromptTemplate：聊天消息模板（推荐）
# =========================================================

print("\n" + "=" * 60)
print("练习 2：ChatPromptTemplate 基本使用")
print("=" * 60)

# 方式 1（推荐）：from_messages()
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role}，回答要 {style}。"),
    ("human", "{question}"),
])

print(f"模板变量: {chat_prompt.input_variables}")

# 三种调用方式
# 方式 1：format() → 返回字符串
print("\n--- format() → 返回字符串 ---")
result = chat_prompt.format(
    role="Python导师",
    style="简洁明了",
    question="什么是装饰器？",
)
print(result)

# 方式 2：format_messages() → 返回消息列表（推荐）
print("\n--- format_messages() → 返回消息列表 ---")
messages = chat_prompt.format_messages(
    role="Python导师",
    style="简洁明了",
    question="什么是装饰器？",
)
for m in messages:
    print(f"  {type(m).__name__}: {m.content}")

# 方式 3：invoke() → 返回 ChatPromptValue（最灵活）
print("\n--- invoke() → 返回 ChatPromptValue ---")
result = chat_prompt.invoke({
    "role": "Python导师",
    "style": "简洁明了",
    "question": "什么是装饰器？",
})
print(f"  类型: {type(result).__name__}")
print(f"  转消息: {result.to_messages()}")


# =========================================================
# 3. 更丰富的初始化参数类型
# =========================================================

print("\n" + "=" * 60)
print("练习 3：多种参数类型")
print("=" * 60)

# 类型 1：字符串列表
prompt_str = ChatPromptTemplate.from_messages([
    "你是一个翻译助手",  # 字符串默认为 human 角色
    "把以下内容翻译成英文：{text}",
])
print("类型1 - 字符串列表:")
for m in prompt_str.format_messages(text="你好世界"):
    print(f"  {type(m).__name__}: {m.content}")

# 类型 2：字典列表
prompt_dict = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "你是数据分析助手。"},
    {"role": "human", "content": "分析这组数据：{data}"},
])
print("\n类型2 - 字典列表:")
for m in prompt_dict.format_messages(data="[1,2,3,4,5]"):
    print(f"  {type(m).__name__}: {m.content}")

# 类型 3：Message 对象列表
prompt_msg = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是助手。"),
    HumanMessage(content="你好！"),  # 不带变量的固定消息
])
print("\n类型3 - Message对象列表:")
for m in prompt_msg.format_messages():
    print(f"  {type(m).__name__}: {m.content}")


# =========================================================
# 4. partial()：部分变量预填充
# =========================================================

print("\n" + "=" * 60)
print("练习 4：partial() 部分变量预填充")
print("=" * 60)

# 原始模板有 3 个变量
original_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role}，擅长 {skill}。"),
    ("human", "{question}"),
])

# 预填充部分变量，创建模板变体
python_tutor = original_prompt.partial(role="Python导师", skill="Python编程")
# 现在只需要提供 question
result = python_tutor.format(question="如何读取文件？")
print(f"预填充后的模板:")
print(result)

# 可以创建多个变体
data_analyst = original_prompt.partial(role="数据分析师", skill="数据分析")
print(f"\n另一个变体:")
print(data_analyst.format(question="如何处理缺失值？"))


# =========================================================
# 5. MessagesPlaceholder：消息占位符
# =========================================================

print("\n" + "=" * 60)
print("练习 5：MessagesPlaceholder 消息占位符")
print("=" * 60)

# 场景：需要在模板中插入一组动态消息（如对话历史）
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 对话历史占位符
    ("human", "{question}"),
])

# 准备对话历史
history = [
    HumanMessage(content="我叫张三。"),
    AIMessage(content="你好张三！"),
    HumanMessage(content="我在学Python。"),
    AIMessage(content="太好了！Python很有趣。"),
]

# 使用模板（插入历史 + 新问题）
messages = prompt_with_history.format_messages(
    chat_history=history,
    question="根据之前的对话，我叫什么？在学什么？",
)

print(f"生成的消息列表（{len(messages)} 条）:")
for m in messages:
    print(f"  {type(m).__name__}: {m.content}")


# =========================================================
# 6. 可复用模板库
# =========================================================

print("\n" + "=" * 60)
print("练习 6：可复用模板库")
print("=" * 60)

# 在实际项目中，建议创建模板库统一管理


class PromptLibrary:
    """可复用的提示词模板库"""

    # 翻译模板
    translate = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业翻译，将 {source_lang} 翻译为 {target_lang}。"),
        ("human", "{text}"),
    ])

    # 总结模板
    summarize = ChatPromptTemplate.from_messages([
        ("system", "你是一个文本总结助手，请用 {length} 的篇幅总结。"),
        ("human", "{text}"),
    ])

    # 代码审查模板
    code_review = ChatPromptTemplate.from_messages([
        ("system", "你是一个 {language} 代码审查专家。"),
        ("human", "请审查以下代码，指出问题和改进建议：\n```\n{code}\n```"),
    ])

    # 问答模板（带历史）
    qa_with_history = ChatPromptTemplate.from_messages([
        ("system", "你是 {role}。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


# 使用模板库
print("--- 翻译模板 ---")
msgs = PromptLibrary.translate.format_messages(
    source_lang="中文", target_lang="英文", text="今天天气真好"
)
for m in msgs:
    print(f"  {type(m).__name__}: {m.content}")

print("\n--- 代码审查模板 ---")
msgs = PromptLibrary.code_review.format_messages(
    language="Python",
    code="def add(a, b): return a+b",
)
for m in msgs:
    print(f"  {type(m).__name__}: {m.content}")


# =========================================================
# 7. 模板组合
# =========================================================

print("\n" + "=" * 60)
print("练习 7：模板组合")
print("=" * 60)

# 将多个模板片段组合成复杂的提示词

# 片段 1：系统设定
system_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role}。"),
])

# 片段 2：任务说明
task_template = ChatPromptTemplate.from_messages([
    ("system", "你的任务是 {task}。"),
])

# 片段 3：用户输入
input_template = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
])

# 组合模板
combined = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role}。"),
    ("system", "你的任务是 {task}。"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{input}"),
])

# 使用组合模板
messages = combined.format_messages(
    role="Python导师",
    task="回答编程问题并给出代码示例",
    history=[],  # 可以传入历史消息
    input="如何写一个装饰器？",
)

print(f"组合后的消息（{len(messages)} 条）:")
for m in messages:
    print(f"  {type(m).__name__}: {m.content}")


# =========================================================
# 8. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 04 章提示词模板部分的核心知识点：

  1. PromptTemplate：基础字符串模板，输出纯文本
  2. ChatPromptTemplate：聊天消息模板，支持角色（推荐）
  3. from_messages()：推荐的创建方式
  4. 三种调用方式：
     - format()        → 返回字符串
     - format_messages() → 返回消息列表
     - invoke()        → 返回 ChatPromptValue（最灵活）
  5. partial()：预填充部分变量，创建模板变体
  6. MessagesPlaceholder：插入动态消息列表（如对话历史）
  7. 可复用模板库：统一管理项目中的提示词
  8. 模板组合：将多个片段组合成复杂提示词

PromptTemplate vs ChatPromptTemplate：
  特性          | PromptTemplate | ChatPromptTemplate
  --------------|---------------|-------------------
  输出格式      | 纯文本字符串   | 消息列表
  角色支持      | ❌            | ✅ system/user/assistant
  推荐          | 简单场景      | 开发中使用（推荐）

结合 LLM 使用：
  chain = prompt | model  →  先格式化模板，再传给模型
""")
