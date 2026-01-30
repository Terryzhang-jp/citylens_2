"""
CityLens V7 - Hybrid Pipeline + ReAct Discovery

核心改进：
- Layer 1/2/4 保持不变（Pipeline结构清晰）
- Layer 3 引入 ReAct Discovery Loop（动态追问）

Layer 3 改进：
- 从"预设并行搜索"改为"种子并行 + 动态深挖"
- 引入 surprise_score 判断发现价值
- 设置 budget 防止无限循环
- 动态追问：发现引出新问题时继续探索

流程图：
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: Triage (保持不变)                                         │
│  "有意思吗？" → none/surface/deep                                   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2: Parallel Observation (保持不变)                           │
│  多视角并行观察 → 提取"研究种子"                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
╔═════════════════════════════════════════════════════════════════════╗
║  Layer 3: ReAct Discovery Loop 【核心改动】                         ║
║                                                                     ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │ Phase 1: 并行搜索所有种子                                    │   ║
║  │ seeds: ["独特入口设计", "屋顶植被", ...]                     │   ║
║  │         ↓ 并行                                              │   ║
║  │ initial_discoveries: [{fact, surprise_score, followup}, ...] │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║                               │                                     ║
║                               ▼                                     ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │ Phase 2: 动态深挖 (ReAct Loop)                               │   ║
║  │                                                             │   ║
║  │ while budget > 0 and has_followup:                          │   ║
║  │     THOUGHT: 哪个发现值得追问？                              │   ║
║  │     ACTION:  搜索 followup 问题                              │   ║
║  │     OBSERVE: 分析结果，评估 surprise_score                   │   ║
║  │     DECIDE:  新发现是否产生更多问题？                         │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║                                                                     ║
╚═════════════════════════════════════════════════════════════════════╝
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4: Synthesize (保持不变)                                     │
│  基于所有发现生成洞见                                               │
└─────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import time
import os
from typing import Optional, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel

from google import genai
from google.genai import types

from src.utils.llm import parse_json_response
from src.utils.location import get_nearby_pois, format_pois_for_prompt
from src.agents.director import PERSPECTIVE_POOL
from src.utils.logger import llm_logger, analysis_logger


# ============================================================
# 配置
# ============================================================

MODEL = "gemini-3-flash-preview"
MAX_DISCOVERY_BUDGET = 5  # 最多搜索轮数
MIN_SURPRISE_FOR_FOLLOWUP = 0.6  # 超过此分数才深挖


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ResearchSeed:
    """从 Surface Analysis 提取的研究种子"""
    observation: str      # "入口使用了大量镜面材料"
    hypothesis: str       # "可能是某种特殊设计手法"
    perspective: str      # 来源视角
    priority: float = 0.5 # 优先级 0-1


@dataclass
class Discovery:
    """一次搜索的发现"""
    query: str                    # 搜索词
    fact: str                     # 核心事实
    detail: str                   # 详细内容
    source_summary: str           # 来源摘要
    surprise_score: float         # 惊人程度 0-1
    followup: Optional[str] = None  # 产生的新问题
    depth: int = 0                # 搜索深度 (0=初始, 1=追问, 2=再追问...)


@dataclass
class PlannerDecision:
    """Planner 的决策"""
    action: str           # "search" | "done"
    query: Optional[str]  # 如果 search，搜什么
    reasoning: str        # 为什么这样决定


# ============================================================
# 进度展示
# ============================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class ProgressDisplay:
    def __init__(self):
        self.start_time = time.time()

    def header(self, text: str):
        print(f"\n{Colors.BOLD}{'═' * 70}")
        print(f"  {text}")
        print(f"{'═' * 70}{Colors.ENDC}")

    def layer(self, num: int, name: str, desc: str):
        elapsed = time.time() - self.start_time
        print(f"\n{Colors.CYAN}┌{'─' * 68}┐{Colors.ENDC}")
        print(f"{Colors.CYAN}│{Colors.BOLD} Layer {num}: {name} {Colors.ENDC}{Colors.DIM}@ {elapsed:.1f}s{Colors.ENDC}")
        print(f"{Colors.CYAN}│{Colors.ENDC} {desc}")
        print(f"{Colors.CYAN}└{'─' * 68}┘{Colors.ENDC}")

    def phase(self, name: str):
        print(f"\n  {Colors.BOLD}▸ {name}{Colors.ENDC}")

    def task(self, name: str, result: str = "", status: str = "success"):
        icon = {"success": "✓", "error": "✗", "pending": "○", "thinking": "?"}.get(status, "·")
        color = {"success": Colors.GREEN, "error": Colors.RED, "thinking": Colors.YELLOW}.get(status, Colors.DIM)
        result_str = f" → {result}" if result else ""
        print(f"    {color}{icon}{Colors.ENDC} {name}{result_str}")

    def react_thought(self, thought: str):
        print(f"    {Colors.YELLOW}💭 THOUGHT:{Colors.ENDC} {thought}")

    def react_action(self, action: str):
        print(f"    {Colors.BLUE}🔍 ACTION:{Colors.ENDC} {action}")

    def react_observe(self, observation: str, score: float):
        score_bar = "★" * int(score * 5) + "☆" * (5 - int(score * 5))
        print(f"    {Colors.GREEN}👁 OBSERVE:{Colors.ENDC} {observation}")
        print(f"             {Colors.DIM}surprise: {score_bar} ({score:.2f}){Colors.ENDC}")

    def react_decide(self, decision: str):
        print(f"    {Colors.CYAN}→ DECIDE:{Colors.ENDC} {decision}")

    def discovery(self, d: Discovery):
        depth_marker = "└" + "─" * d.depth + "▸" if d.depth > 0 else "▸"
        score_display = f"[{'★' * int(d.surprise_score * 5)}{'☆' * (5 - int(d.surprise_score * 5))}]"
        print(f"    {Colors.GREEN}{depth_marker}{Colors.ENDC} {d.fact[:60]}... {Colors.DIM}{score_display}{Colors.ENDC}")

    def timing(self, timings: dict):
        print(f"\n{Colors.BOLD}{'─' * 70}")
        print(f"  ⏱️  耗时总结")
        print(f"{'─' * 70}{Colors.ENDC}")
        total = timings.get("total", 1)
        for name, t in sorted(timings.items(), key=lambda x: -x[1]):
            if name == "total":
                continue
            pct = (t / total * 100)
            bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
            print(f"  {name:20s} {bar} {t:5.1f}s ({pct:4.1f}%)")
        print(f"{'─' * 70}")
        print(f"  {Colors.BOLD}总计: {total:.1f}s{Colors.ENDC}")


progress = ProgressDisplay()


# ============================================================
# 异步 LLM 工具
# ============================================================

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("请设置 GEMINI_API_KEY 环境变量")
    return genai.Client(api_key=api_key)


class LLMError(Exception):
    """LLM API 调用错误"""
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


async def llm_with_image(
    prompt: str,
    image_data: bytes,
    json_mode: bool = False,
    max_retries: int = 2,
    timeout_seconds: float = 60.0,
) -> str:
    """
    调用 Gemini API 分析图片，带错误处理和重试

    Args:
        prompt: 提示词
        image_data: 图片数据
        json_mode: 是否返回 JSON
        max_retries: 最大重试次数
        timeout_seconds: 超时时间

    Returns:
        模型输出文本

    Raises:
        LLMError: API 调用失败
    """
    client = get_client()
    config = types.GenerateContentConfig(temperature=0.5, max_output_tokens=4096)
    if json_mode:
        config.response_mime_type = "application/json"

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # 带超时的 API 调用
            async with asyncio.timeout(timeout_seconds):
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=[types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    ])],
                    config=config,
                )

            # 检查空响应
            if not response.text:
                llm_logger.warning(f"LLM 返回空响应 (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # 递增延迟
                    continue
                raise LLMError("LLM 返回空响应", retryable=True)

            return response.text

        except asyncio.TimeoutError:
            last_error = LLMError(f"API 调用超时 ({timeout_seconds}s)", retryable=True)
            llm_logger.warning(f"LLM 超时 (attempt {attempt + 1}/{max_retries + 1})")

        except Exception as e:
            error_str = str(e).lower()
            # 判断是否可重试
            retryable = any(x in error_str for x in ["429", "500", "503", "timeout", "rate"])

            if "401" in error_str or "invalid" in error_str:
                # 认证错误，不重试
                llm_logger.error(f"LLM 认证错误: {e}")
                raise LLMError(f"API 认证失败: {e}", retryable=False)

            last_error = LLMError(str(e), retryable=retryable)
            llm_logger.warning(f"LLM 错误 (attempt {attempt + 1}): {e}")

        # 重试前等待
        if attempt < max_retries:
            wait_time = 2.0 * (attempt + 1)  # 2s, 4s, 6s...
            llm_logger.info(f"等待 {wait_time}s 后重试...")
            await asyncio.sleep(wait_time)

    # 所有重试都失败
    llm_logger.error(f"LLM 调用失败，已重试 {max_retries} 次: {last_error}")
    raise last_error or LLMError("未知错误")


async def llm_text(
    prompt: str,
    json_mode: bool = False,
    temperature: float = 0.5,
    max_retries: int = 2,
    timeout_seconds: float = 45.0,
) -> str:
    """
    调用 Gemini API（纯文本），带错误处理和重试

    Args:
        prompt: 提示词
        json_mode: 是否返回 JSON
        temperature: 温度参数
        max_retries: 最大重试次数
        timeout_seconds: 超时时间

    Returns:
        模型输出文本

    Raises:
        LLMError: API 调用失败
    """
    client = get_client()
    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=4096)
    if json_mode:
        config.response_mime_type = "application/json"

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                    config=config,
                )

            if not response.text:
                llm_logger.warning(f"LLM text 返回空响应 (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError("LLM 返回空响应", retryable=True)

            return response.text

        except asyncio.TimeoutError:
            last_error = LLMError(f"API 调用超时 ({timeout_seconds}s)", retryable=True)
            llm_logger.warning(f"LLM text 超时 (attempt {attempt + 1})")

        except LLMError:
            raise  # 直接传递 LLMError

        except Exception as e:
            error_str = str(e).lower()
            retryable = any(x in error_str for x in ["429", "500", "503", "timeout", "rate"])
            last_error = LLMError(str(e), retryable=retryable)
            llm_logger.warning(f"LLM text 错误 (attempt {attempt + 1}): {e}")

        if attempt < max_retries:
            wait_time = 2.0 * (attempt + 1)
            await asyncio.sleep(wait_time)

    llm_logger.error(f"LLM text 调用失败: {last_error}")
    raise last_error or LLMError("未知错误")


async def search_grounding(query: str, context: str = "", timeout_seconds: float = 30.0) -> dict:
    """
    使用 Google Search 进行搜索，带超时和错误处理

    Args:
        query: 搜索查询
        context: 背景上下文
        timeout_seconds: 超时时间

    Returns:
        {"answer": str, "sources": list, "error": Optional[str]}
    """
    client = get_client()
    prompt = f"""请搜索并回答：{query}
{f"背景：{context}" if context else ""}
提供准确、有深度的回答，包含具体细节。用中文回答。"""

    try:
        async with asyncio.timeout(timeout_seconds):
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            if metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        sources.append(getattr(chunk.web, 'title', ''))

        answer = response.text or ""
        if not answer:
            llm_logger.warning(f"搜索返回空结果: {query[:50]}...")

        return {"answer": answer, "sources": sources}

    except asyncio.TimeoutError:
        llm_logger.warning(f"搜索超时 ({timeout_seconds}s): {query[:50]}...")
        return {"answer": "搜索超时，请稍后重试", "sources": [], "error": "timeout"}

    except Exception as e:
        llm_logger.error(f"搜索失败: {e}")
        return {"answer": "搜索暂时不可用", "sources": [], "error": str(e)}


# ============================================================
# 状态定义
# ============================================================

@dataclass
class AgentState:
    image_data: bytes = b""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_context: str = ""

    # 用户输入（Plan B）
    user_description: str = ""  # 用户对照片的描述
    photo_type: str = "auto"    # auto/building/abstract/other
    has_markup: bool = False    # 图片是否包含用户标记的好奇区域

    # Layer 1
    perception: str = ""
    interest_level: str = "none"
    interest_reason: str = ""
    needs_location: bool = True  # Plan A: 是否需要位置信息
    user_specified_location: str = ""  # 用户明确指定的地点（从 user_description 提取）
    suggested_perspectives: list = field(default_factory=list)

    # Layer 2
    research_seeds: list = field(default_factory=list)  # List[ResearchSeed]
    surface_findings: list = field(default_factory=list)

    # Layer 3 - ReAct Discovery
    discoveries: list = field(default_factory=list)  # List[Discovery]
    discovery_budget_used: int = 0
    react_trace: list = field(default_factory=list)  # 记录 ReAct 过程

    # Layer 4
    final_response: dict = field(default_factory=dict)
    response_type: str = "nothing"

    timings: dict = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================
# Layer 1: Triage (保持不变)
# ============================================================

async def layer1_triage(state: AgentState) -> AgentState:
    progress.layer(1, "Triage", "判断这张照片是否有值得探索的内容")

    # 构建用户输入上下文（Plan B）
    user_context = ""
    if state.user_description:
        user_context += f"\n## 用户描述\n{state.user_description}"
    if state.photo_type != "auto":
        type_hints = {
            "building": "用户表示这是建筑/城市相关照片",
            "abstract": "用户表示这是抽象/艺术性照片，可能与具体地点无关",
            "other": "用户未指定照片类型",
        }
        user_context += f"\n## 照片类型提示\n{type_hints.get(state.photo_type, '')}"

    # 用户标记提示
    markup_context = ""
    if state.has_markup:
        markup_context = """
## 重要：用户标记区域
图片中带有半透明黄色/橙色标记的区域是用户特别感兴趣的部分。
请优先关注这些标记区域，它们应该被视为 "deep" 级别的兴趣点。
即使图片整体平淡，只要有标记区域就应该深入分析。
"""

    prompt = f"""你是一位好奇的观察者。

## 核心原则（必须遵守）
- **只描述你在图片中实际看到的内容**
- **绝对禁止**猜测、推断或编造图片中不存在的元素
- 如果不确定某个元素是什么，说"不确定"而非猜测
- 宁可遗漏，不可编造

## 任务
看这张图片，判断：是否有任何值得深入了解的东西？
{markup_context}{user_context}
{f"## 位置信息{chr(10)}{state.location_context}" if state.location_context else ""}

## 判断标准
- "none": 普通日常场景，没有独特元素
- "surface": 有具体可识别的事物，表面有趣
- "deep": 可识别的特定建筑/地点/作品，值得深挖

## 是否需要位置信息 (needs_location)
判断这张照片是否需要通过地理位置来增强分析：
- true: 照片包含建筑、店铺、街道、地标等，位置信息能帮助识别具体地点
- false: 照片是抽象的（光影、纹理、艺术构图）、或主体与地点无关（食物特写、产品、自然景观细节）

## 可用视角
{chr(10).join([f"- {pid}: {p['name']}" for pid, p in PERSPECTIVE_POOL.items()])}

## 输出 JSON
{{
    "perception": "简短描述你实际看到的内容（1句话）",
    "interest_level": "none/surface/deep",
    "interest_reason": "判断原因（1句话）",
    "needs_location": true/false,
    "location_reason": "为什么需要/不需要位置信息（1句话）",
    "user_specified_location": "如果用户在描述中明确提到了地点名称，提取出来；否则为空字符串",
    "suggested_perspectives": ["perspective_id", ...]
}}

**关于 user_specified_location**：
- 如果用户说"地点在秩父神社"，则提取"秩父神社"
- 如果用户说"这是东京塔附近拍的"，则提取"东京塔"
- 如果用户没有提到具体地点，则填空字符串 ""
"""

    try:
        response = await llm_with_image(prompt, state.image_data, json_mode=True)
        result = parse_json_response(response, {})

        # 确保 result 是 dict
        if not isinstance(result, dict):
            result = {}

        state.perception = result.get("perception", "")
        state.interest_level = result.get("interest_level", "none")
        state.interest_reason = result.get("interest_reason", "")
        state.needs_location = result.get("needs_location", True)  # 默认需要位置
        state.user_specified_location = result.get("user_specified_location", "")
        state.suggested_perspectives = result.get("suggested_perspectives", [])

        # 如果用户明确指定了 abstract 类型，强制不需要位置
        if state.photo_type == "abstract":
            state.needs_location = False

        level_display = {"none": "无特别发现", "surface": "表面有趣", "deep": "值得深挖"}
        progress.task("分析图片", state.perception)
        progress.task("判断结果", f"{level_display.get(state.interest_level, '?')} - {state.interest_reason}")

        location_status = "需要" if state.needs_location else "不需要"
        location_reason = result.get("location_reason", "")
        progress.task("位置信息", f"{location_status} - {location_reason}")

        # 显示用户指定的地点
        if state.user_specified_location:
            progress.task("用户指定地点", f"✓ {state.user_specified_location}")

    except Exception as e:
        state.error = str(e)
        state.interest_level = "none"

    return state


# ============================================================
# Layer 2: Parallel Observation (改进：输出研究种子)
# ============================================================

async def observe_one_perspective(
    image_data: bytes,
    perspective_id: str,
    perception: str,
    location_context: str,
    user_specified_location: str = "",
    has_markup: bool = False,
) -> tuple[list[dict], list[ResearchSeed]]:
    """观察并提取研究种子"""

    perspective = PERSPECTIVE_POOL.get(perspective_id, {})
    name = perspective.get("name", perspective_id)
    expertise = perspective.get("expertise", "")

    # 构建地点约束
    location_constraint = ""
    if user_specified_location:
        location_constraint = f"""
## 重要：用户已确认地点
用户明确告知这是「{user_specified_location}」。
- **不要猜测其他可能的地点**
- 所有分析和假设都应基于这是{user_specified_location}的前提
- 搜索关键词应包含「{user_specified_location}」
"""

    # 用户标记提示
    markup_constraint = ""
    if has_markup:
        markup_constraint = """
## 重要：用户标记区域
图片中带有半透明黄色/橙色标记的区域是用户特别感兴趣的部分。
- **优先分析标记区域**，这是用户最想了解的内容
- 为标记区域生成高优先级的研究种子 (priority >= 0.8)
- 如果能识别标记区域的具体内容，将其作为主要发现
"""

    prompt = f"""你是一位{name}，专长于{expertise}。

## 核心原则（必须遵守）
- **只观察和描述图片中实际存在的元素**
- **绝对禁止**编造、想象或推测图片中不存在的内容
- 每个发现必须对应图片中**可见的**具体区域或元素
- 如果从你的视角看不到值得注意的内容，返回空的 findings 数组
- 宁可少报告，不可编造
{location_constraint}{markup_constraint}
## 背景
图片内容：{perception}
{f"位置：{location_context}" if location_context else ""}

## 任务
从你的专业视角观察图片，找出**实际可见的**值得探索的发现。
{f"注意：地点已确定为{user_specified_location}，请围绕此地点进行分析。" if user_specified_location else ""}

## 输出 JSON
{{
    "findings": [
        {{
            "name": "发现名称（必须是图中可见的具体事物）",
            "observation": "客观观察（只描述你看到的）",
            "insight": "专业解读",
            "bounding_box": {{
                "x1": 150,
                "y1": 200,
                "x2": 450,
                "y2": 600
            }},
            "research_seed": {{
                "hypothesis": "这可能是什么？需要搜索验证的假设",
                "search_query": "建议的搜索关键词{f'（应包含{user_specified_location}）' if user_specified_location else ''}",
                "priority": 0.8  // 优先级 0-1
            }}
        }}
    ]
}}

**bounding_box 说明**:
- 标注发现对应的图片区域，使用归一化坐标 (0-1000)
- x1,y1 是左上角，x2,y2 是右下角
- 0 表示最左/最上，1000 表示最右/最下
- 例如：左上四分之一区域 = {{"x1":0,"y1":0,"x2":500,"y2":500}}

**重要**：
- 最多 1-2 个发现
- 只报告图片中**清晰可见**的内容
- 如果看不清或不确定，不要报告
- **必须为每个发现提供 bounding_box**，标注该发现在图片中的位置
{f"- 地点已确认是{user_specified_location}，不要猜测其他地点" if user_specified_location else "- 如果不确定地点，可以基于视觉特征提出假设"}
"""

    try:
        response = await llm_with_image(prompt, image_data, json_mode=True)
        result = parse_json_response(response, {})
        findings = result.get("findings", [])

        seeds = []
        for f in findings:
            f["perspective_id"] = perspective_id
            f["perspective_name"] = name

            if f.get("research_seed"):
                seed_data = f["research_seed"]
                seeds.append(ResearchSeed(
                    observation=f.get("observation", ""),
                    hypothesis=seed_data.get("hypothesis", seed_data.get("search_query", "")),
                    perspective=name,
                    priority=seed_data.get("priority", 0.5),
                ))

        return findings, seeds
    except Exception as e:
        return [], []


async def layer2_observation(state: AgentState) -> AgentState:
    progress.layer(2, "Parallel Observation", "多视角并行观察，提取研究种子")

    perspectives = [p for p in state.suggested_perspectives if p in PERSPECTIVE_POOL]
    if not perspectives:
        perspectives = ["architect", "storyteller"]

    # 限制视角数量以优化性能
    perspectives = perspectives[:2]

    progress.phase(f"并行观察 ({len(perspectives)} 个视角)")

    # 如果用户指定了地点，显示提示
    if state.user_specified_location:
        progress.task("用户指定地点", f"✓ {state.user_specified_location}（将围绕此地点分析）")

    tasks = [
        observe_one_perspective(
            state.image_data, p, state.perception, state.location_context,
            state.user_specified_location,  # 传递用户指定的地点
            state.has_markup,  # 传递是否有用户标记
        )
        for p in perspectives
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_findings = []
    all_seeds = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            progress.task(f"[{perspectives[i]}]", str(result), "error")
            continue

        findings, seeds = result
        all_findings.extend(findings)
        all_seeds.extend(seeds)

        for f in findings:
            has_seed = "🔬" if f.get("research_seed") else "👁"
            progress.task(f"[{perspectives[i]}]", f"{has_seed} {f.get('name', '?')}")

    state.surface_findings = all_findings
    state.research_seeds = all_seeds

    progress.phase(f"提取研究种子: {len(all_seeds)} 个")
    for seed in all_seeds:
        print(f"    {Colors.DIM}· [{seed.perspective}] {seed.hypothesis[:50]}... (priority: {seed.priority}){Colors.ENDC}")

    return state


# ============================================================
# Layer 3: ReAct Discovery Loop 【核心改动】
# ============================================================

async def analyze_search_result(query: str, search_result: dict, context: str) -> Discovery:
    """分析搜索结果，评估 surprise_score，判断是否有 followup"""

    prompt = f"""分析搜索结果，提取与照片相关的发现。

## 搜索词
{query}

## 搜索结果
{search_result.get('answer', '')[:2000]}

## 照片上下文
{context}

## 评估标准
- surprise_score: 这个发现对普通人来说有多惊人？(0-1)
  - 0.0-0.3: 常识，大家都知道
  - 0.4-0.6: 有点意思，但不算惊人
  - 0.7-0.9: 很惊人，"原来如此！"
  - 1.0: 极其惊人，改变认知

- followup: 这个发现是否引出更深的问题？
  - 如果发现了具体的名称/人物/事件，可以追问细节
  - 如果只是泛泛的信息，不需要追问

## 输出 JSON
{{
    "fact": "核心事实（1句话，最重要的发现）",
    "detail": "详细内容（2-3句话）",
    "surprise_score": 0.75,
    "followup": "引出的新问题（如无则为null）",
    "followup_reason": "为什么要追问这个问题"
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True)
        result = parse_json_response(response, {})

        # 确保 result 是 dict
        if not isinstance(result, dict):
            result = {}

        # 处理 sources - 可能是 dict 列表或 str 列表
        sources = search_result.get("sources", [])
        if sources and isinstance(sources[0], dict):
            source_names = [s.get("title", "") for s in sources[:3]]
        else:
            source_names = sources[:3]

        return Discovery(
            query=query,
            fact=result.get("fact", "未知"),
            detail=result.get("detail", ""),
            source_summary=", ".join(filter(None, source_names)),
            surprise_score=float(result.get("surprise_score", 0.5)),
            followup=result.get("followup") if result.get("followup") else None,
        )
    except Exception as e:
        return Discovery(
            query=query,
            fact=f"分析失败: {e}",
            detail="",
            source_summary="",
            surprise_score=0.0,
        )


async def planner_decide(
    context: str,
    discoveries: list[Discovery],
    pending_questions: list[str],
    budget_remaining: int,
) -> PlannerDecision:
    """Planner 决定下一步：继续搜索还是结束"""

    discoveries_text = "\n".join([
        f"- [{d.surprise_score:.1f}] {d.fact}" + (f" → 追问: {d.followup}" if d.followup else "")
        for d in discoveries
    ]) if discoveries else "（暂无）"

    questions_text = "\n".join([f"- {q}" for q in pending_questions]) if pending_questions else "（暂无）"

    prompt = f"""你是 CityLens 的研究规划者。

## 照片上下文
{context}

## 已有发现
{discoveries_text}

## 待探索问题
{questions_text}

## 剩余搜索次数
{budget_remaining}

## 决策规则
1. 如果已有 2-3 个高质量发现（surprise_score > 0.7），可以结束
2. 如果待探索问题中有明显值得追问的，继续搜索
3. 如果剩余次数少且已有足够发现，结束
4. 优先追问能挖出"冷知识"的问题

## 输出 JSON
{{
    "action": "search" 或 "done",
    "query": "如果 search，具体搜什么（精确的搜索词）",
    "reasoning": "为什么这样决定（1句话）"
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True, temperature=0.3)
        result = parse_json_response(response, {})

        return PlannerDecision(
            action=result.get("action", "done"),
            query=result.get("query"),
            reasoning=result.get("reasoning", ""),
        )
    except Exception as e:
        return PlannerDecision(action="done", query=None, reasoning=f"决策失败: {e}")


async def layer3_react_discovery(state: AgentState) -> AgentState:
    """
    Layer 3: ReAct Discovery Loop

    Phase 1: 并行搜索所有初始种子
    Phase 2: 动态追问有价值的发现
    """
    progress.layer(3, "ReAct Discovery", "种子并行搜索 + 动态追问")

    seeds = state.research_seeds
    if not seeds:
        progress.task("跳过", "没有研究种子", "pending")
        return state

    # 构建搜索上下文，包含用户指定的地点
    context = f"{state.perception}. {state.interest_reason}"
    if state.user_specified_location:
        context = f"地点：{state.user_specified_location}。{context}"
        progress.task("搜索上下文", f"将围绕「{state.user_specified_location}」进行搜索")

    discoveries: list[Discovery] = []
    react_trace = []
    budget = MAX_DISCOVERY_BUDGET

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: 并行搜索初始种子
    # ═══════════════════════════════════════════════════════════════

    # 按优先级排序，取前3个（优化性能）
    sorted_seeds = sorted(seeds, key=lambda s: s.priority, reverse=True)[:3]

    progress.phase(f"Phase 1: 并行搜索初始种子 ({len(sorted_seeds)} 个)")

    async def search_and_analyze(seed: ResearchSeed) -> Discovery:
        # 如果用户指定了地点，将其加入搜索词
        search_query = seed.hypothesis
        if state.user_specified_location and state.user_specified_location not in search_query:
            search_query = f"{state.user_specified_location} {search_query}"

        search_result = await search_grounding(search_query, seed.observation)
        discovery = await analyze_search_result(search_query, search_result, context)
        discovery.depth = 0
        return discovery

    search_tasks = [search_and_analyze(s) for s in sorted_seeds]
    initial_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for i, result in enumerate(initial_results):
        if isinstance(result, Exception):
            progress.task(f"搜索 [{sorted_seeds[i].perspective}]", str(result), "error")
        else:
            discoveries.append(result)
            progress.discovery(result)
            react_trace.append({
                "phase": 1,
                "type": "initial_search",
                "query": result.query,
                "surprise_score": result.surprise_score,
                "has_followup": result.followup is not None,
            })

    budget -= len(sorted_seeds)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: 动态追问 (ReAct Loop)
    # ═══════════════════════════════════════════════════════════════
    progress.phase("Phase 2: 动态追问 (ReAct Loop)")

    # 收集所有待追问的问题
    pending_questions = [
        d.followup for d in discoveries
        if d.followup and d.surprise_score >= MIN_SURPRISE_FOR_FOLLOWUP
    ]

    # 检查 Phase 1 是否已有足够高质量发现
    high_value_count = sum(1 for d in discoveries if d.surprise_score >= 0.7)
    if high_value_count >= 2:
        progress.task("Phase 1 已有足够发现", f"{high_value_count} 个高价值", "pending")
        pending_questions = []  # 跳过 Phase 2

    iteration = 0
    while budget > 0 and pending_questions and iteration < 2:  # 最多2轮追问
        iteration += 1
        print(f"\n    {Colors.DIM}─── ReAct 迭代 {iteration} (剩余 budget: {budget}) ───{Colors.ENDC}")

        # THOUGHT: Planner 决定下一步
        decision = await planner_decide(context, discoveries, pending_questions, budget)
        progress.react_thought(decision.reasoning)

        react_trace.append({
            "phase": 2,
            "iteration": iteration,
            "type": "thought",
            "decision": decision.action,
            "reasoning": decision.reasoning,
        })

        if decision.action == "done":
            progress.react_decide("发现足够，结束探索")
            break

        # ACTION: 执行搜索
        progress.react_action(f"搜索: {decision.query}")
        search_result = await search_grounding(decision.query, context)

        # OBSERVE: 分析结果
        new_discovery = await analyze_search_result(decision.query, search_result, context)
        new_discovery.depth = iteration
        discoveries.append(new_discovery)

        progress.react_observe(new_discovery.fact, new_discovery.surprise_score)

        react_trace.append({
            "phase": 2,
            "iteration": iteration,
            "type": "discovery",
            "query": new_discovery.query,
            "fact": new_discovery.fact,
            "surprise_score": new_discovery.surprise_score,
        })

        # DECIDE: 更新待追问列表
        if new_discovery.followup and new_discovery.surprise_score >= MIN_SURPRISE_FOR_FOLLOWUP:
            pending_questions.append(new_discovery.followup)
            progress.react_decide(f"新问题加入队列: {new_discovery.followup[:40]}...")
        else:
            progress.react_decide("此线索探索完毕")

        # 移除已探索的问题
        pending_questions = [q for q in pending_questions if q != decision.query]
        budget -= 1

    # 总结
    high_value = [d for d in discoveries if d.surprise_score >= 0.7]
    progress.phase(f"探索完成: {len(discoveries)} 个发现, {len(high_value)} 个高价值")

    state.discoveries = discoveries
    state.discovery_budget_used = MAX_DISCOVERY_BUDGET - budget
    state.react_trace = react_trace

    return state


# ============================================================
# Layer 4: Synthesize (保持不变，但使用 discoveries)
# ============================================================

async def layer4_synthesize_nothing(state: AgentState) -> AgentState:
    progress.layer(4, "Synthesize", "生成响应（无特别发现）")

    state.final_response = {
        "type": "nothing",
        "message": f"这张照片展示了{state.perception}。{state.interest_reason}",
        "suggestion": "试试拍摄一些独特的建筑、有趣的细节吧！",
    }
    state.response_type = "nothing"
    return state


async def layer4_synthesize_surface(state: AgentState) -> AgentState:
    progress.layer(4, "Synthesize", "整合表面观察")

    findings_text = "\n".join([
        f"- [{f.get('perspective_name', '')}] {f.get('name', '')}: {f.get('insight', '')}"
        for f in state.surface_findings
    ])

    prompt = f"""整合观察为简洁有趣的发现。

## 图片
{state.perception}

## 观察
{findings_text}

## 输出 JSON
{{
    "summary": "一句话总结",
    "findings": [{{"title": "...", "content": "..."}}],
    "closing": "引发思考的一句话"
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True)
        result = parse_json_response(response, {})

        state.final_response = {
            "type": "surface",
            "summary": result.get("summary", ""),
            "findings": result.get("findings", []),
            "closing": result.get("closing", ""),
        }
        state.response_type = "surface"

        progress.task("生成完成", f"{len(result.get('findings', []))} 个发现")

    except Exception as e:
        state.error = str(e)

    return state


async def layer4_synthesize_deep(state: AgentState) -> AgentState:
    progress.layer(4, "Synthesize", "生成深度洞见")

    # 按 surprise_score 排序，取最好的发现
    sorted_discoveries = sorted(state.discoveries, key=lambda d: d.surprise_score, reverse=True)

    discoveries_text = "\n".join([
        f"""
### 发现 {i+1} (惊人度: {d.surprise_score:.1f})
- 核心事实: {d.fact}
- 详细内容: {d.detail}
- 搜索深度: {'初始发现' if d.depth == 0 else f'追问第{d.depth}层'}
"""
        for i, d in enumerate(sorted_discoveries[:5])
    ])

    # 也包含表面观察（包含 bounding_box 信息）
    surface_text = "\n".join([
        f"- [{f.get('perspective_name', '')}] {f.get('name', '')}: {f.get('insight', '')} (区域: {f.get('bounding_box', 'N/A')})"
        for f in state.surface_findings
    ])

    # 构建发现名称到 bounding_box 的映射，供合成时使用
    finding_bboxes = {
        f.get('name', ''): f.get('bounding_box')
        for f in state.surface_findings
        if f.get('bounding_box')
    }

    prompt = f"""你是一位知识传播者，善于把专业知识变得有趣易懂。

## 核心原则（必须遵守）
- **所有洞见必须基于图片中实际可见的内容**
- **绝对禁止**编造图片中不存在的元素或细节
- 每个 insight 的 title 必须对应图片中**真实存在**的视觉元素
- 如果研究发现与图片内容不符，忽略该发现
- 宁可少生成洞见，不可编造

## 图片内容
{state.perception}

## 深度研究发现（需验证是否与图片匹配）
{discoveries_text}

## 表面观察
{surface_text}

## 任务
基于这些发现，生成 2-3 条让普通人惊叹的洞见。
**关键**：每个洞见必须能在图片中找到对应的视觉证据。

## 输出 JSON
{{
    "insights": [
        {{
            "title": "标题（必须对应图片中可见的具体元素）",
            "hook": "开场钩子（引起好奇的问题）",
            "explanation": "核心解读（3-4句话，把发现讲清楚）",
            "visual_evidence": "这个洞见对应图片中的哪个可见元素",
            "source_finding": "对应的表面观察名称（用于匹配区域坐标）",
            "fun_fact": "冷知识（如果有的话，必须与图片相关）"
        }}
    ],
    "theme": "主题总结（1句话）",
    "invitation": "邀请探索的一句话"
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True, temperature=0.7)
        result = parse_json_response(response, {})

        # 为每个 insight 匹配 bounding_box
        insights = result.get("insights", [])
        for insight in insights:
            source_name = insight.get("source_finding", "")
            # 尝试精确匹配
            if source_name and source_name in finding_bboxes:
                insight["bounding_box"] = finding_bboxes[source_name]
            else:
                # 尝试模糊匹配（名称包含关系）
                for name, bbox in finding_bboxes.items():
                    if name in source_name or source_name in name:
                        insight["bounding_box"] = bbox
                        break

        state.final_response = {
            "type": "deep",
            "insights": insights,
            "theme": result.get("theme", ""),
            "invitation": result.get("invitation", ""),
        }
        state.response_type = "deep"

        progress.task("生成完成", f"{len(insights)} 条洞见")

        for insight in insights:
            bbox_status = "📍" if insight.get("bounding_box") else "❌"
            print(f"    {Colors.GREEN}💡{Colors.ENDC} {Colors.BOLD}{insight.get('title', '')}{Colors.ENDC} {bbox_status}")

    except Exception as e:
        state.error = str(e)

    return state


# ============================================================
# 主流程
# ============================================================

async def run_perception_v7(
    image_data: bytes,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    user_description: str = "",
    photo_type: str = "auto",
) -> dict:
    """V7 主流程

    Args:
        image_data: 图片数据
        latitude: 纬度（可选）
        longitude: 经度（可选）
        user_description: 用户对照片的描述（Plan B）
        photo_type: 照片类型 auto/building/abstract/other（Plan B）
    """

    total_start = time.time()
    progress.header("🧪 CityLens V7 - Hybrid Pipeline + ReAct Discovery")

    state = AgentState(
        image_data=image_data,
        latitude=latitude,
        longitude=longitude,
        user_description=user_description,
        photo_type=photo_type,
    )

    timings = {}

    # Layer 1: Triage（先判断，再决定是否获取位置）
    t1 = time.time()
    state = await layer1_triage(state)
    timings["triage"] = time.time() - t1

    if state.interest_level == "none":
        state = await layer4_synthesize_nothing(state)
        timings["total"] = time.time() - total_start
        state.timings = timings
        progress.timing(timings)
        return _format_result(state)

    # Location（Plan A: 仅在需要时获取）
    if latitude and longitude and state.needs_location:
        loc_start = time.time()
        print(f"\n  {Colors.DIM}📍 获取位置信息 ({latitude}, {longitude})...{Colors.ENDC}")
        loop = asyncio.get_event_loop()
        state.location_context = await loop.run_in_executor(
            None, lambda: format_pois_for_prompt(get_nearby_pois(latitude, longitude, 100))
        )
        timings["location"] = time.time() - loc_start
    elif latitude and longitude and not state.needs_location:
        print(f"\n  {Colors.DIM}📍 跳过位置信息（照片内容与地点无关）{Colors.ENDC}")
        timings["location"] = 0.0

    # Layer 2: Observation
    t2 = time.time()
    state = await layer2_observation(state)
    timings["observation"] = time.time() - t2

    if not state.research_seeds:
        state = await layer4_synthesize_surface(state)
        timings["total"] = time.time() - total_start
        state.timings = timings
        progress.timing(timings)
        return _format_result(state)

    # Layer 3: ReAct Discovery
    t3 = time.time()
    state = await layer3_react_discovery(state)
    timings["react_discovery"] = time.time() - t3

    # Layer 4: Synthesize
    t4 = time.time()
    state = await layer4_synthesize_deep(state)
    timings["synthesize"] = time.time() - t4

    timings["total"] = time.time() - total_start
    state.timings = timings

    progress.timing(timings)

    return _format_result(state)


def _format_result(state: AgentState) -> dict:
    return {
        "type": state.response_type,
        "perception": state.perception,
        "response": state.final_response,
        "timings": state.timings,
        "error": state.error,
        "process": {
            "interest_level": state.interest_level,
            "needs_location": state.needs_location,  # Plan A: 是否使用了位置信息
            "user_specified_location": state.user_specified_location,  # 用户指定的地点
            "seeds_count": len(state.research_seeds),
            "discoveries_count": len(state.discoveries),
            "budget_used": state.discovery_budget_used,
            "react_trace": state.react_trace,
        },
        "user_input": {  # Plan B: 用户输入
            "description": state.user_description,
            "photo_type": state.photo_type,
        }
    }


# ============================================================
# 流式进度版本 (SSE)
# ============================================================

async def run_perception_v7_streaming(
    image_data: bytes,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    user_description: str = "",
    photo_type: str = "auto",
    has_markup: bool = False,
):
    """流式版本 - 通过 yield 返回进度事件

    Args:
        has_markup: 如果为 True，表示图片包含用户涂抹的标记区域，
                   分析时会优先关注这些区域
    """

    total_start = time.time()

    yield {
        "type": "progress",
        "layer": 0,
        "phase": "start",
        "message": "开始分析...",
    }

    state = AgentState(
        image_data=image_data,
        latitude=latitude,
        longitude=longitude,
        user_description=user_description,
        photo_type=photo_type,
        has_markup=has_markup,
    )

    timings = {}

    # Layer 1: Triage
    triage_detail = "分析这张照片是否有值得探索的内容"
    if has_markup:
        triage_detail = "分析用户标记的感兴趣区域"

    yield {
        "type": "progress",
        "layer": 1,
        "phase": "triage",
        "message": "判断图片内容...",
        "detail": triage_detail,
    }

    t1 = time.time()
    state = await layer1_triage(state)
    timings["triage"] = time.time() - t1

    yield {
        "type": "progress",
        "layer": 1,
        "phase": "triage_done",
        "message": f"初步判断: {state.perception[:50]}..." if state.perception else "分析完成",
        "detail": f"兴趣级别: {state.interest_level}",
    }

    if state.interest_level == "none":
        state = await layer4_synthesize_nothing(state)
        timings["total"] = time.time() - total_start
        state.timings = timings
        yield {"type": "result", "success": True, "data": _format_result(state)}
        return

    # Location
    if latitude and longitude and state.needs_location:
        yield {
            "type": "progress",
            "layer": 1,
            "phase": "location",
            "message": f"获取位置信息...",
            "detail": f"坐标: {latitude:.4f}, {longitude:.4f}",
        }
        loc_start = time.time()
        loop = asyncio.get_event_loop()
        state.location_context = await loop.run_in_executor(
            None, lambda: format_pois_for_prompt(get_nearby_pois(latitude, longitude, 100))
        )
        timings["location"] = time.time() - loc_start

    # Layer 2: Observation
    # 获取实际使用的视角名称
    perspectives_to_use = [p for p in state.suggested_perspectives if p in PERSPECTIVE_POOL]
    if not perspectives_to_use:
        perspectives_to_use = ["architect", "storyteller"]
    perspectives_to_use = perspectives_to_use[:2]  # 限制2个

    perspective_names = [PERSPECTIVE_POOL[p]["name"] for p in perspectives_to_use]
    perspectives_detail = "、".join(perspective_names) + "视角并行分析"

    yield {
        "type": "progress",
        "layer": 2,
        "phase": "observation",
        "message": "多视角观察中...",
        "detail": perspectives_detail,
    }

    t2 = time.time()
    state = await layer2_observation(state)
    timings["observation"] = time.time() - t2

    seeds_count = len(state.research_seeds)
    yield {
        "type": "progress",
        "layer": 2,
        "phase": "observation_done",
        "message": f"发现 {seeds_count} 个研究线索",
        "detail": ", ".join([s.observation[:20] + "..." for s in state.research_seeds[:3]]) if state.research_seeds else "",
    }

    if not state.research_seeds:
        state = await layer4_synthesize_surface(state)

        # 为 surface findings 添加分割
        findings_with_bbox = [
            f for f in state.surface_findings
            if f.get("bounding_box")
        ]

        if findings_with_bbox:
            yield {
                "type": "progress",
                "layer": 5,
                "phase": "segmentation",
                "message": f"提取关键区域...",
                "detail": f"为 {len(findings_with_bbox)} 个发现生成抠图",
            }

            try:
                from src.utils.segmentation import segment_region

                for finding in state.surface_findings:
                    bbox = finding.get("bounding_box")
                    if bbox and isinstance(bbox, dict):
                        if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                            cropped_image = segment_region(
                                state.image_data,
                                bbox,
                                output_format="png",
                            )
                            if cropped_image:
                                finding["cropped_image"] = cropped_image

                # 将带抠图的 findings 添加到 final_response
                state.final_response["surface_findings"] = state.surface_findings
            except Exception as e:
                print(f"[Segmentation] Surface 分割失败: {e}")

        timings["total"] = time.time() - total_start
        state.timings = timings
        yield {"type": "result", "success": True, "data": _format_result(state)}
        return

    # Layer 3: ReAct Discovery
    yield {
        "type": "progress",
        "layer": 3,
        "phase": "discovery",
        "message": "深入搜索研究中...",
        "detail": "通过网络搜索验证和扩展发现",
    }

    t3 = time.time()
    state = await layer3_react_discovery(state)
    timings["react_discovery"] = time.time() - t3

    discoveries_count = len(state.discoveries)
    high_value = len([d for d in state.discoveries if d.surprise_score >= 0.7])
    yield {
        "type": "progress",
        "layer": 3,
        "phase": "discovery_done",
        "message": f"完成 {discoveries_count} 个发现",
        "detail": f"其中 {high_value} 个高价值发现",
    }

    # Layer 4: Synthesize
    yield {
        "type": "progress",
        "layer": 4,
        "phase": "synthesize",
        "message": "生成洞见报告...",
        "detail": "整合所有发现，生成深度分析",
    }

    t4 = time.time()
    state = await layer4_synthesize_deep(state)
    timings["synthesize"] = time.time() - t4

    timings["total"] = time.time() - total_start
    state.timings = timings

    insights_count = len(state.final_response.get("insights", [])) if state.final_response else 0
    yield {
        "type": "progress",
        "layer": 4,
        "phase": "complete",
        "message": f"分析完成！生成 {insights_count} 条洞见",
        "detail": f"总耗时: {timings['total']:.1f}秒",
    }

    # Layer 5: Segmentation (为有 bounding_box 的 insight 生成抠图)
    if state.final_response and state.final_response.get("insights"):
        insights_with_bbox = [
            i for i in state.final_response["insights"]
            if i.get("bounding_box")
        ]

        if insights_with_bbox:
            yield {
                "type": "progress",
                "layer": 5,
                "phase": "segmentation",
                "message": f"提取关键区域...",
                "detail": f"为 {len(insights_with_bbox)} 个洞见生成抠图",
            }

            t5 = time.time()
            try:
                from src.utils.segmentation import segment_region

                for insight in state.final_response["insights"]:
                    bbox = insight.get("bounding_box")
                    if bbox and isinstance(bbox, dict):
                        # 确保 bbox 有所有必需的键
                        if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                            cropped_image = segment_region(
                                state.image_data,
                                bbox,
                                output_format="png",
                            )
                            if cropped_image:
                                insight["cropped_image"] = cropped_image

                timings["segmentation"] = time.time() - t5

                # 统计成功抠图数量
                cropped_count = len([
                    i for i in state.final_response["insights"]
                    if i.get("cropped_image")
                ])

                yield {
                    "type": "progress",
                    "layer": 5,
                    "phase": "segmentation_done",
                    "message": f"区域提取完成",
                    "detail": f"成功提取 {cropped_count}/{len(insights_with_bbox)} 个区域",
                }
            except Exception as e:
                print(f"[Segmentation] 分割步骤失败: {e}")
                timings["segmentation"] = time.time() - t5
                yield {
                    "type": "progress",
                    "layer": 5,
                    "phase": "segmentation_error",
                    "message": "区域提取跳过",
                    "detail": str(e)[:50],
                }

    # Update total time after segmentation
    timings["total"] = time.time() - total_start
    state.timings = timings

    # Final result
    yield {"type": "result", "success": True, "data": _format_result(state)}


# 同步入口
def enhance_perception_v7(
    image_data: bytes,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    user_description: str = "",
    photo_type: str = "auto",
) -> dict:
    return asyncio.run(run_perception_v7(
        image_data, latitude, longitude, user_description, photo_type
    ))


# ============================================================
# Layer 5: Curiosity Expansion (好奇心扩展)
# ============================================================

@dataclass
class MindmapNode:
    """Mindmap 节点"""
    id: str
    title: str
    content: str
    node_type: str = "branch"  # root/branch/leaf
    children: list = field(default_factory=list)
    expanded: bool = False
    depth: int = 0


async def generate_curiosity_questions(insights: list[dict], perception: str) -> list[dict]:
    """
    从洞见中生成好奇心问题

    Returns:
        list of {"question": str, "topic": str, "context": str}
    """
    insights_text = "\n".join([
        f"- {i.get('title', '')}: {i.get('explanation', '')[:100]}..."
        for i in insights
    ])

    prompt = f"""基于以下分析洞见，生成用户可能感兴趣的探索问题。

## 图片内容
{perception}

## 分析洞见
{insights_text}

## 任务
提取洞见中提到的具体名词、人物、概念、现象，生成 4-6 个引导用户深入探索的问题。

## 问题设计原则
1. 具体化：针对具体事物（如"墨西哥飞蓬是什么？"而非"这花是什么？"）
2. 故事性：挖掘背后的人物/历史（如"XX的创始人是谁？"）
3. 关联性：探索事物之间的联系（如"为什么XX会出现在YY？"）
4. 深度性：引导思考更深层原因（如"为什么这种设计在日本流行？"）

## 输出 JSON
{{
    "questions": [
        {{
            "question": "完整的问题句子",
            "topic": "核心主题词（2-4字）",
            "context": "为什么这个问题有趣（1句话）",
            "keywords": ["搜索关键词1", "搜索关键词2"]
        }}
    ]
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True)
        result = parse_json_response(response, {})
        if not isinstance(result, dict):
            return []
        return result.get("questions", [])
    except Exception as e:
        print(f"生成问题失败: {e}")
        return []


async def generate_mindmap(question: str, topic: str, context: str, keywords: list[str]) -> dict:
    """
    为一个问题生成层层递进的 Mindmap 结构

    Returns:
        Mindmap structure with root and branches
    """
    # 先搜索获取信息
    search_query = " ".join(keywords) if keywords else topic
    search_result = await search_grounding(search_query, question)

    prompt = f"""基于搜索结果，为用户构建一个知识 Mindmap。

## 用户问题
{question}

## 主题
{topic}

## 搜索结果
{search_result.get('answer', '')[:2000]}

## 任务
构建一个层层递进的知识结构，帮助用户建立心智模型。

## Mindmap 设计原则
1. 根节点：核心概念的简洁定义
2. 一级分支：3-4个关键维度（是什么/为什么/怎么样/关联）
3. 二级分支：每个维度下2-3个具体要点
4. 每个节点都应该简洁有力，像知识卡片

## 输出 JSON
{{
    "root": {{
        "title": "核心概念名称",
        "summary": "一句话核心定义",
        "emoji": "合适的emoji"
    }},
    "branches": [
        {{
            "id": "branch_1",
            "title": "分支标题（如：起源）",
            "emoji": "🌱",
            "summary": "2-3句话概述",
            "key_points": [
                {{
                    "title": "要点标题",
                    "content": "具体内容（1-2句话）",
                    "expandable": true,
                    "expand_query": "如果用户想深入，应该搜索什么"
                }}
            ]
        }}
    ],
    "fun_fact": "一个有趣的冷知识",
    "related_questions": ["延伸问题1", "延伸问题2"]
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True, temperature=0.6)
        result = parse_json_response(response, {})
        if not isinstance(result, dict):
            return {"error": "解析失败", "raw": str(response)[:200]}

        # 验证必要字段
        if not result.get("root"):
            # 尝试构建默认结构
            result["root"] = {
                "title": topic or question[:20],
                "summary": search_result.get("answer", "")[:100] + "...",
                "emoji": "📚"
            }

        if not result.get("branches"):
            result["branches"] = [{
                "id": "branch_1",
                "title": "基本信息",
                "emoji": "📌",
                "summary": search_result.get("answer", "")[:200],
                "key_points": []
            }]

        result["search_sources"] = search_result.get("sources", [])
        return result
    except Exception as e:
        return {"error": str(e), "search_answer": search_result.get("answer", "")[:300]}


async def expand_branch(branch_title: str, expand_query: str, parent_context: str) -> dict:
    """
    展开 Mindmap 的某个分支，获取更深入的内容
    """
    search_result = await search_grounding(expand_query, parent_context)

    prompt = f"""用户想深入了解 Mindmap 中的某个分支。

## 分支主题
{branch_title}

## 上下文
{parent_context}

## 搜索结果
{search_result.get('answer', '')[:1500]}

## 任务
提供这个分支的深入解读，包括：
1. 更详细的解释
2. 具体的例子或案例
3. 为什么这很重要
4. 可能的进一步探索方向

## 输出 JSON
{{
    "deep_explanation": "详细解释（3-5句话）",
    "examples": ["具体例子1", "具体例子2"],
    "significance": "为什么这很重要（1-2句话）",
    "go_deeper": [
        {{
            "direction": "探索方向",
            "query": "搜索词"
        }}
    ]
}}
"""

    try:
        response = await llm_text(prompt, json_mode=True, temperature=0.5)
        result = parse_json_response(response, {})
        if not isinstance(result, dict):
            return {"error": "解析失败"}
        return result
    except Exception as e:
        return {"error": str(e)}


# 同步入口
def generate_curiosity_questions_sync(insights: list[dict], perception: str) -> list[dict]:
    """同步版本：生成好奇心问题"""
    return asyncio.run(generate_curiosity_questions(insights, perception))


def generate_mindmap_sync(question: str, topic: str, context: str, keywords: list[str]) -> dict:
    """同步版本：生成 Mindmap"""
    return asyncio.run(generate_mindmap(question, topic, context, keywords))


def expand_branch_sync(branch_title: str, expand_query: str, parent_context: str) -> dict:
    """同步版本：展开分支"""
    return asyncio.run(expand_branch(branch_title, expand_query, parent_context))
