"""
LLM 客户端工具
"""

import os
import json
import re
from google import genai
from google.genai import types


# 模型配置
MODEL_FLASH_LITE = "gemini-2.5-flash-lite"
MODEL_FLASH = "gemini-2.5-flash"
MODEL_3_FLASH = "gemini-3-flash-preview"  # 最新模型

# 默认使用的模型
DEFAULT_MODEL = MODEL_3_FLASH


def get_client() -> genai.Client:
    """获取 Gemini 客户端"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("请设置 GEMINI_API_KEY 环境变量")
    return genai.Client(api_key=api_key)


def call_gemini_with_image(
    prompt: str,
    image_data: bytes,
    mime_type: str = "image/jpeg",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    调用 Gemini API 分析图片

    Args:
        prompt: 提示词
        image_data: 图片二进制数据
        mime_type: 图片 MIME 类型
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大输出 token 数
        json_mode: 是否启用 JSON 模式

    Returns:
        模型输出文本
    """
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if json_mode:
        config.response_mime_type = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                ],
            ),
        ],
        config=config,
    )

    return response.text


def call_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    调用 Gemini API（纯文本）

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大输出 token 数
        json_mode: 是否启用 JSON 模式

    Returns:
        模型输出文本
    """
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if json_mode:
        config.response_mime_type = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ],
        config=config,
    )

    return response.text


def call_gemini_with_search(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 4096,
) -> str:
    """
    调用 Gemini API 并启用 Google Search

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大输出 token 数

    Returns:
        模型输出文本
    """
    client = get_client()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    return response.text


def parse_json_response(text: str, default: dict | list = None) -> dict | list:
    """
    解析 LLM 返回的 JSON

    Args:
        text: LLM 输出文本
        default: 解析失败时的默认值

    Returns:
        解析后的 JSON 对象
    """
    if default is None:
        default = {}

    if not text or not text.strip():
        return default

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 对象边界
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 数组边界
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    print(f"[JSON Parse] Failed to parse: {text[:200]}...")
    return default
