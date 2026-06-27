"""
第 06 章 结构化输出 练习 2：with_structured_output 四种模式
=================================================
本练习需要大模型 API，让 AI 返回结构化的 Python 对象。
学习目标：
  1. with_structured_output() 核心方法
  2. 四种定义模式：Pydantic / TypedDict / JSON Schema / @dataclass
  3. include_raw=True 获取原始响应
  4. 实际应用场景：信息提取、情感分析、自动分类

前置条件：
  1. 将 configs/.env.example 复制为 configs/.env
  2. 填入你的 API Key（推荐 DeepSeek）

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/llm_basics/04_结构化输出_四种模式.py"
"""

import os
import json
from enum import Enum
from typing import Optional, Literal, List
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


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
print(f"✅ 模型初始化成功")


# =========================================================
# 2. 模式 1：Pydantic BaseModel（最推荐）
# =========================================================

print("\n" + "=" * 60)
print("模式 1：Pydantic BaseModel")
print("=" * 60)


class MovieExtraction(BaseModel):
    """从文本中提取电影信息"""
    title: str = Field(description="电影的标题")
    year: int = Field(description="电影的上映年份")
    director: str = Field(description="电影的导演")
    rating: Optional[float] = Field(default=None, description="电影评分，0-10分")
    genre: List[str] = Field(description="电影类型列表，如科幻、动作等")


# 使用 with_structured_output 绑定结构
structured_model = model.with_structured_output(MovieExtraction)

# 让大模型从自然语言文本中提取结构化信息
text = "《流浪地球2》是2023年上映的一部科幻电影，由郭帆执导，豆瓣评分8.3分，属于科幻、灾难类型。"
print(f"输入文本: {text}")

result = structured_model.invoke(text)
print(f"\n返回类型: {type(result)}")
print(f"标题: {result.title}")
print(f"年份: {result.year}")
print(f"导演: {result.director}")
print(f"评分: {result.rating}")
print(f"类型: {result.genre}")


# =========================================================
# 3. 模式 2：TypedDict
# =========================================================

print("\n" + "=" * 60)
print("模式 2：TypedDict")
print("=" * 60)

from typing_extensions import TypedDict, Annotated


class BookInfo(TypedDict):
    """书籍信息"""
    title: Annotated[str, ..., "书名"]
    author: Annotated[str, ..., "作者"]
    pages: Annotated[int, ..., "页数"]
    category: Annotated[Literal["小说", "技术", "历史", "其他"], "分类"]


structured_model_2 = model.with_structured_output(BookInfo)

text2 = "《Python编程：从入门到实践》由Eric Matthes编写，共564页，是一本技术类书籍。"
print(f"输入文本: {text2}")

result2 = structured_model_2.invoke(text2)
print(f"\n返回类型: {type(result2)}")
print(f"结果: {result2}")
print(f"书名: {result2['title']}, 作者: {result2['author']}")


# =========================================================
# 4. 模式 3：JSON Schema
# =========================================================

print("\n" + "=" * 60)
print("模式 3：JSON Schema")
print("=" * 60)

json_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["正面", "负面", "中性"],
            "description": "情感分析结果",
        },
        "confidence": {
            "type": "number",
            "description": "置信度，0-1之间",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "关键情感词列表",
        },
    },
    "required": ["sentiment", "confidence", "keywords"],
}

structured_model_3 = model.with_structured_output(json_schema)

text3 = "这个产品真的太棒了！质量很好，物流也快，非常满意！强烈推荐！"
print(f"输入文本: {text3}")

result3 = structured_model_3.invoke(text3)
print(f"\n返回类型: {type(result3)}")
print(f"情感: {result3['sentiment']}")
print(f"置信度: {result3['confidence']}")
print(f"关键词: {result3['keywords']}")


# =========================================================
# 5. 模式 4：@dataclass
# =========================================================

print("\n" + "=" * 60)
print("模式 4：@dataclass")
print("=" * 60)


@dataclass
class PersonInfo:
    """人物信息"""
    name: str  # 姓名
    age: int   # 年龄
    occupation: str  # 职业
    city: Optional[str]  # 城市


structured_model_4 = model.with_structured_output(PersonInfo)

text4 = "张三今年28岁，是一名软件工程师，目前在北京工作。"
print(f"输入文本: {text4}")

result4 = structured_model_4.invoke(text4)
print(f"\n返回类型: {type(result4)}")
print(f"结果: {result4}")


# =========================================================
# 6. include_raw=True：获取原始响应
# =========================================================

print("\n" + "=" * 60)
print("进阶：include_raw=True")
print("=" * 60)

structured_model_raw = model.with_structured_output(
    MovieExtraction,
    include_raw=True,
)

text5 = "《肖申克的救赎》是1994年上映的经典剧情片，导演是弗兰克·德拉邦特，豆瓣评分9.7。"
print(f"输入文本: {text5}")

result5 = structured_model_raw.invoke(text5)
print(f"\n返回类型: {type(result5)}")
print(f"包含字段: {list(result5.keys())}")
print(f"\n[raw] 原始AIMessage:")
print(f"  {result5['raw'].content[:200]}...")
print(f"\n[parsed] 解析后的对象:")
parsed = result5['parsed']
print(f"  标题: {parsed.title}")
print(f"  年份: {parsed.year}")
print(f"  导演: {parsed.director}")
print(f"  评分: {parsed.rating}")
print(f"\n[errors] 解析错误: {result5['errors']}")


# =========================================================
# 7. 实战应用：批量信息提取
# =========================================================

print("\n" + "=" * 60)
print("实战应用：批量信息提取")
print("=" * 60)


class OrderInfo(BaseModel):
    """订单信息提取"""
    order_id: str = Field(description="订单编号")
    product: str = Field(description="商品名称")
    quantity: int = Field(description="购买数量")
    price: float = Field(description="单价")
    status: Literal["待付款", "已付款", "已发货", "已完成", "已取消"] = Field(
        description="订单状态"
    )
    customer_note: Optional[str] = Field(
        default=None, description="客户备注（可选）"
    )


structured_order = model.with_structured_output(OrderInfo)

orders_text = [
    "订单DD20240115001，买了3件iPhone 15 Pro，每件8999元，已经付款了，麻烦尽快发货。",
    "订单DD20240115002，1台MacBook Pro，单价18999元，等待付款中。",
    "订单DD20240115003，2双运动鞋，每双399元，已经收到了，非常满意！请帮我备注：下次多买几双。",
]

for text in orders_text:
    print(f"\n输入: {text}")
    order = structured_order.invoke(text)
    print(f"  订单号: {order.order_id}")
    print(f"  商品: {order.product} x{order.quantity}")
    print(f"  单价: ¥{order.price}")
    print(f"  状态: {order.status}")
    if order.customer_note:
        print(f"  备注: {order.customer_note}")


# =========================================================
# 8. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
结构化输出四种模式对比：

  模式           | 返回类型   | 校验  | 推荐度
  ---------------|-----------|-------|-------
  Pydantic       | 对象      | ✅    | ⭐⭐⭐⭐⭐
  TypedDict      | 字典      | ❌    | ⭐⭐⭐
  JSON Schema    | 字典      | ❌    | ⭐⭐⭐
  @dataclass     | 字典      | ❌    | ⭐⭐

最佳实践：
  1. 优先使用 Pydantic BaseModel（有类型校验，返回可直接点属性）
  2. 复杂嵌套结构建议拆分成多个调用
  3. 使用 include_raw=True 调试时查看原始响应
  4. 用 Field(description=...) 帮助大模型理解每个字段
  5. 用 Literal / Enum 限制枚举值
  6. 用 Optional 标记可选字段
""")
