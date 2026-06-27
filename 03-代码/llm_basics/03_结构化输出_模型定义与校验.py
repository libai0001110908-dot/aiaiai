"""
第 06 章 结构化输出 练习 1：Pydantic 模型定义与校验
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 使用 Pydantic BaseModel 定义结构化输出模型
  2. 可选字段（Optional）
  3. 枚举类型（Enum / Literal）
  4. 嵌套结构
  5. 字段校验与限制条件
  6. JSON 序列化

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/llm_basics/03_结构化输出_模型定义与校验.py"
"""

import json
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# =========================================================
# 1. 基本使用：定义结构化输出模型
# =========================================================

print("=" * 60)
print("练习 1：基本的 Pydantic 模型定义")
print("=" * 60)


class MovieInfo(BaseModel):
    """电影信息结构化模型"""
    title: str = Field(description="电影的标题")
    year: int = Field(description="电影的上映年份")
    director: str = Field(description="电影的导演")
    rating: float = Field(description="电影评分，0-10分")


# 模拟大模型返回的结构化数据
movie = MovieInfo(
    title="流浪地球2",
    year=2023,
    director="郭帆",
    rating=8.3,
)

print(f"标题: {movie.title}")
print(f"年份: {movie.year}")
print(f"导演: {movie.director}")
print(f"评分: {movie.rating}")

# 转换为字典 / JSON
print(f"\n字典形式: {movie.model_dump()}")
print(f"JSON 形式: {movie.model_dump_json()}")


# =========================================================
# 2. 可选字段（Optional）
# =========================================================

print("\n" + "=" * 60)
print("练习 2：可选字段（Optional）")
print("=" * 60)


class UserProfile(BaseModel):
    """用户资料模型"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄")
    email: Optional[str] = Field(default=None, description="邮箱地址（可选）")
    phone: Optional[str] = Field(default=None, description="手机号（可选）")


# 情况1：只填必填字段
user1 = UserProfile(name="张三", age=25)
print(f"用户1: {user1.model_dump()}")

# 情况2：填写所有字段
user2 = UserProfile(name="李四", age=30, email="lisi@example.com", phone="13800138000")
print(f"用户2: {user2.model_dump()}")


# =========================================================
# 3. 枚举类型（Enum / Literal）
# =========================================================

print("\n" + "=" * 60)
print("练习 3：枚举类型（Enum / Literal）")
print("=" * 60)


# 方式1：使用 Enum
class PriorityLevel(Enum):
    """工单优先级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    URGENT = "紧急"


class SupportTicket(BaseModel):
    """客服工单模型"""
    title: str = Field(description="工单标题")
    priority: PriorityLevel = Field(description="优先级")
    description: str = Field(description="问题描述")


ticket = SupportTicket(
    title="无法登录账号",
    priority=PriorityLevel.HIGH,
    description="用户反馈输入正确密码但无法登录",
)
print(f"工单: {ticket.model_dump()}")
print(f"优先级: {ticket.priority.value}")


# 方式2：使用 Literal（更简洁）
class NewsArticle(BaseModel):
    """新闻文章模型"""
    title: str = Field(description="新闻标题")
    category: Literal["科技", "财经", "体育", "娱乐", "政治"] = Field(
        description="新闻分类"
    )
    summary: str = Field(description="新闻摘要")


news = NewsArticle(
    title="OpenAI 发布新模型",
    category="科技",
    summary="OpenAI 今日发布了新一代大语言模型...",
)
print(f"\n新闻: {news.model_dump()}")


# =========================================================
# 4. 嵌套结构
# =========================================================

print("\n" + "=" * 60)
print("练习 4：嵌套结构")
print("=" * 60)


class Address(BaseModel):
    """地址模型"""
    city: str = Field(description="城市")
    street: str = Field(description="街道地址")
    zip_code: Optional[str] = Field(default=None, description="邮编")


class Company(BaseModel):
    """公司模型（包含嵌套的地址）"""
    name: str = Field(description="公司名称")
    industry: str = Field(description="所属行业")
    address: Address = Field(description="公司地址")
    employee_count: int = Field(description="员工数量")


company = Company(
    name="尚硅谷",
    industry="IT教育",
    address=Address(
        city="北京",
        street="海淀区中关村大街1号",
        zip_code="100080",
    ),
    employee_count=500,
)

print(f"公司: {company.model_dump_json(indent=2, ensure_ascii=False)}")


# =========================================================
# 5. 字段校验与限制条件
# =========================================================

print("\n" + "=" * 60)
print("练习 5：字段校验与限制条件")
print("=" * 60)


class ProductReview(BaseModel):
    """商品评价模型"""
    product_name: str = Field(
        min_length=1, max_length=100,
        description="商品名称"
    )
    rating: float = Field(
        ge=0, le=5,  # 限制在 0-5 之间
        description="评分，0-5分"
    )
    review_text: str = Field(
        max_length=500,
        description="评价内容"
    )

    @field_validator("review_text")
    @classmethod
    def review_not_empty(cls, v):
        if not v.strip():
            raise ValueError("评价内容不能为空")
        return v.strip()


# 正常情况
review1 = ProductReview(
    product_name="iPhone 15",
    rating=4.5,
    review_text="很好用，拍照清晰！",
)
print(f"评价1: {review1.model_dump()}")

# 校验失败的情况
print("\n--- 测试校验失败 ---")
try:
    review2 = ProductReview(
        product_name="测试商品",
        rating=6.0,  # 超出范围
        review_text="",
    )
except Exception as e:
    print(f"校验失败（预期内）: {e}")


# =========================================================
# 6. 查看模型 JSON Schema（大模型通过它理解输出格式）
# =========================================================

print("\n" + "=" * 60)
print("练习 6：查看模型的 JSON Schema")
print("=" * 60)

print("SupportTicket 的 JSON Schema:")
print(json.dumps(
    SupportTicket.model_json_schema(),
    ensure_ascii=False, indent=2
))


# =========================================================
# 7. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 06 章的核心知识点：
  1. Pydantic BaseModel：定义结构化输出模型的基础
  2. Optional：标记可选字段
  3. Enum / Literal：限制字段的可选值
  4. 嵌套结构：模型中包含其他模型
  5. Field 约束 + @field_validator：字段校验与限制
  6. model_json_schema()：生成大模型能理解的格式描述

下一步：运行 04_结构化输出_四种模式.py，
        让大模型实际返回结构化数据！
""")
