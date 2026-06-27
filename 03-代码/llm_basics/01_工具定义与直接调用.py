"""
第 05 章 Tools 练习 1：工具的定义与直接调用
=================================================
本练习不需要大模型 API，可直接运行。
学习目标：
  1. 使用 @tool 装饰器定义工具
  2. 直接调用工具（.invoke()）
  3. 查看工具的 schema（名称、描述、参数定义）
  4. 使用 Pydantic 定义 args_schema
  5. 使用 JSON Schema 定义参数模式

运行方式：
    cd d:/workspace/aiaiiai
    .\\venv\\Scripts\\Activate.ps1
    python "03-代码/llm_basics/01_工具定义与直接调用.py"
"""

import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# =========================================================
# 1. 使用 @tool 装饰器定义工具（最常用方式）
# =========================================================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称，如 "北京"、"上海"

    Returns:
        天气信息字符串
    """
    # 模拟实现（实际项目中这里会调用天气 API）
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 22°C",
        "广州": "小雨，温度 28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式，如 "2 + 3"、"10 * 5"

    Returns:
        计算结果字符串
    """
    try:
        result = eval(expression)  # 仅用于演示，生产环境请用 ast.literal_eval
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# =========================================================
# 2. 直接调用工具（方式 1：适合测试）
# =========================================================

print("=" * 60)
print("练习 1：直接调用工具")
print("=" * 60)

# 使用 .invoke() 方法调用
result1 = get_weather.invoke({"city": "北京"})
print(f"get_weather('北京') => {result1}")

result2 = get_weather.invoke({"city": "上海"})
print(f"get_weather('上海') => {result2}")

result3 = calculate.invoke({"expression": "2 + 3"})
print(f"calculate('2 + 3') => {result3}")

result4 = calculate.invoke({"expression": "10 * 5"})
print(f"calculate('10 * 5') => {result4}")


# =========================================================
# 3. 查看工具的 Schema（工具描述信息）
# =========================================================

print("\n" + "=" * 60)
print("练习 2：查看工具的 Schema")
print("=" * 60)

# 工具名称
print(f"工具名称: {get_weather.name}")
print(f"工具描述: {get_weather.description}")

# 完整的 tool schema（大模型就是通过这个了解工具的）
print(f"\nget_weather 的完整 schema:")
print(json.dumps(get_weather.tool_call_schema.model_json_schema(),
                 ensure_ascii=False, indent=2))


# =========================================================
# 4. 使用 Pydantic 定义 args_schema（更规范的参数定义）
# =========================================================

print("\n" + "=" * 60)
print("练习 3：使用 Pydantic 定义 args_schema")
print("=" * 60)


class SearchInput(BaseModel):
    """搜索工具的输入参数定义"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大返回结果数量")


@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> str:
    """
    搜索网页内容

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数量

    Returns:
        搜索结果字符串
    """
    # 模拟搜索结果
    return f"搜索 '{query}'，返回 {max_results} 条结果：\n  1. 结果一\n  2. 结果二\n  3. 结果三"


# 调用
result5 = search_web.invoke({"query": "LangChain 教程", "max_results": 3})
print(result5)

# 查看 schema
print(f"\nsearch_web 的参数 schema:")
print(json.dumps(search_web.tool_call_schema.model_json_schema(),
                 ensure_ascii=False, indent=2))


# =========================================================
# 5. 使用 JSON Schema 定义参数模式
# =========================================================

print("\n" + "=" * 60)
print("练习 4：使用 JSON Schema 定义参数模式")
print("=" * 60)

json_schema = {
    "type": "object",
    "properties": {
        "user_id": {
            "type": "string",
            "description": "用户的唯一标识符",
        },
        "include_details": {
            "type": "boolean",
            "description": "是否包含详细信息",
            "default": False,
        },
    },
    "required": ["user_id"],
}


@tool(args_schema=json_schema)
def get_user_info(user_id: str, include_details: bool = False) -> str:
    """获取用户信息"""
    import json as json_lib
    user = {"id": user_id, "name": "张三", "age": 25}
    if include_details:
        user["email"] = "zhangsan@example.com"
        user["address"] = "北京市朝阳区"
    return json_lib.dumps(user, ensure_ascii=False)


result6 = get_user_info.invoke({"user_id": "u001", "include_details": True})
print(f"get_user_info('u001', include_details=True) => {result6}")


# =========================================================
# 6. 练习小结
# =========================================================

print("\n" + "=" * 60)
print("练习小结")
print("=" * 60)
print("""
本练习涵盖了第 05 章的核心知识点：
  1. @tool 装饰器：定义工具的最常用方式
  2. .invoke()：直接调用工具（适合测试）
  3. tool_call_schema：工具的完整描述（大模型通过它理解工具）
  4. Pydantic args_schema：规范化的参数定义方式
  5. JSON Schema：灵活的参数定义方式

下一步：运行 02_工具绑定到模型.py，让 AI 自己决定何时调用工具！
""")
