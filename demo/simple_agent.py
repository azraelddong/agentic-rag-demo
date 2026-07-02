"""
简易 LangChain Agent 示例（含工具调用防错策略）
=================================================

使用 ``langchain.agents.create_agent`` 构建，展示如何防止 agent 调错／漏调工具。

运行方式::

    uv run python demo/simple_agent.py

防错策略总览
------------

===========  ===========================================================  ==========
层级          说明                                                        实现位置
===========  ===========================================================  ==========
① 工具描述    精确的 docstring：做什么、何时用、何时不用、输入示例        每个 @tool
② 系统提示词   路由规则 + Few-shot 示例 + 边界歧义消解规则                SYSTEM_PROMPT
③ 参数校验    使用 Annotated + Field 让 LLM 得到更精确的参数 schema      工具参数定义
④ 中间件      拦截每次 tool_call，打日志、做基础校验（可观察 + 可干预）   ToolMonitorMiddleware
⑤ 容错返回    出错时返回引导性错误信息，帮助 LLM 自我纠正                 每个工具的 except 分支
===========  ===========================================================  ==========
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Annotated, Any, Callable

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

# 确保项目根目录在 sys.path 中 —— 直接运行 demo/simple_agent.py 时
# Python 会把 sys.path[0] 设为脚本所在目录 demo/，导致 app 模块找不到
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# 记忆模块（Redis 会话记忆 + Gatekeeper 结构化条目）
# ---------------------------------------------------------------------------
try:
    from app.core.memory import (
        ConversationMemory, RedisSessionStore,
        MemoryGatekeeper, MemoryClassifier, TurnToolInfo,
    )
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False
    MemoryGatekeeper = None  # type: ignore[assignment,misc]
    MemoryClassifier = None  # type: ignore[assignment,misc]
    TurnToolInfo = None       # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)-5s | %(message)s")
logger = logging.getLogger("agent_demo")


# ---------------------------------------------------------------------------
# 1. 加载环境变量
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ---------------------------------------------------------------------------
# 2. 创建 LLM 实例
# ---------------------------------------------------------------------------
def _build_llm() -> ChatOpenAI:
    """从环境变量构造 ChatOpenAI。"""
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.0,         # ★ 防错关键：temperature=0 减少随机性
        max_tokens=1024,
    )


# ===================================================================
# 3. 定义工具 —— 策略 ①：精确的 docstring（做什么 / 何时用 / 何时不用）
# ===================================================================

@tool
def calculator(
    expression: Annotated[
        str,
        Field(description="数学表达式，如 '2+3*4' 或 'sqrt(16)+cos(0)'"),
    ],
) -> str:
    """执行数学计算。

    ✅ 使用场景：
       - 用户明确要求"计算""算一下""等于多少"的算术题
       - 多步运算、混合运算、带函数的数学表达式
       - 需要精确数值结果的场景

    ❌ 不要使用：
       - 用户只是闲聊数字（如"我今年25岁"）—— 不需要计算
       - 用户问时间 / 天气 / 文本统计 —— 用对应专有工具
       - 用户问常识性问题（如"地球到月球多少公里"）—— 直接回答
       - 纯文本处理（如数字提取、格式化）—— 可能是 word_count 或直接回答

    输入示例: "(15 + 27) * 3"、"sqrt(144)"、"2**10"
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except SyntaxError:
        return (
            f"❌ 表达式语法错误: '{expression}'。请检查括号是否匹配、运算符是否正确。\n"
            f"   正确示例: '2+3*4', '(15+27)*3', 'sqrt(144)'"
        )
    except (ValueError, ZeroDivisionError) as exc:
        return (
            f"❌ 数学错误: {exc}。请检查是否有除零、负数开方等非法操作。\n"
            f"   例如 '1/0' 不允许、'sqrt(-1)' 在实数域无意义。"
        )
    except Exception as exc:
        return f"❌ 计算出错: {exc}"


@tool
def get_current_datetime() -> str:
    """获取当前日期、时间和时区信息。

    ✅ 使用场景：
       - 用户问"现在几点""今天几号""当前时间"
       - 用户需要知道 today 或 now 的准确值

    ❌ 不要使用：
       - 用户问某地天气 —— 用 get_weather（天气工具已包含当地日期）
       - 用户问历史日期计算 —— 用 calculator
       - 用户问某城市的时区但没有提"现在几点" —— 可能只是查城市信息

    无需任何参数。
    """
    now_utc = datetime.now(tz=timezone.utc)
    now_local = datetime.now()
    return (
        f"当前 UTC 时间: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"本地时间: {now_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"本地时区: {now_local.astimezone().tzinfo}"
    )


@tool
def word_count(
    text: Annotated[
        str,
        Field(description="需要统计的文本原文（不要截断或改写）"),
    ],
) -> str:
    """统计一段文本的行数、词数、字符数。

    ✅ 使用场景：
       - 用户给了一段文本，要求统计行/词/字符数
       - 用户问"这段话有多少字"
       - 用户粘贴了内容并说"统计一下"

    ❌ 不要使用：
       - 用户问数学计算 —— 用 calculator
       - 用户要求 JSON 格式化（即使 JSON 很长）—— 用 json_formatter
       - 用户没有给具体文本，只是泛泛地问"怎么统计字数" —— 直接解释方法
       - 对工具返回的结果做二次统计 —— 直接回答即可

    注意: 参数 text 必须是用户提供的原始文本，不要自己生成内容填入。
    """
    lines = text.splitlines()
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", "").replace("\n", ""))
    return (
        f"统计结果:\n"
        f"  - 行数: {len(lines)}\n"
        f"  - 词数: {len(words)}\n"
        f"  - 字符数(含空格): {chars}\n"
        f"  - 字符数(不含空格): {chars_no_spaces}"
    )


@tool
def json_formatter(
    json_string: Annotated[
        str,
        Field(description="原始 JSON 字符串，将对其做格式化（美化缩进）"),
    ],
) -> str:
    """格式化 JSON 字符串，使其缩进对齐、更易阅读。

    ✅ 使用场景：
       - 用户给了原始 JSON 并要求"格式化""美化""整理一下"
       - 用户粘贴了压缩的 JSON 并要求让它可读

    ❌ 不要使用：
       - 用户要求统计 JSON 的字数 —— 用 word_count
       - 用户要求计算 JSON 中的数值 —— 用 calculator（先提取数值）
       - 用户只是给了 JSON 但没有要求格式化 —— 不需要调用
       - 输入根本不是 JSON（纯文本、XML 等）—— 直接告诉用户格式不支持

    注意: 仅处理 JSON 格式。如果用户想格式化其他格式，直接告知不支持。
    """
    try:
        parsed = json.loads(json_string)
        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
        return f"格式化结果:\n{formatted}"
    except json.JSONDecodeError as exc:
        return (
            f"❌ JSON 解析失败: {exc}\n"
            f"   请确认输入是合法的 JSON 字符串。常见错误:\n"
            f"   - 使用了单引号而非双引号\n"
            f"   - 尾部多了逗号\n"
            f"   - key 没有用双引号包裹"
        )


@tool
def get_weather(
    city: Annotated[
        str,
        Field(description="城市名，中文或英文均可。例如: '北京'、'Shanghai'、'Tokyo'"),
    ],
) -> str:
    """查询指定城市当天及未来几天的天气（温度、天气状况、风力等）。

    ✅ 使用场景：
       - 用户问"XX天气怎么样""XX今天多少度""XX会下雨吗"
       - 用户想了解某城市未来几天天气趋势
       - 用户问某地"热不热""冷不冷"等体感类问题

    ❌ 不要使用：
       - 用户问时间/日期 —— 用 get_current_datetime
       - 用户问某城市的人口、面积等非天气信息 —— 直接回答
       - 用户没有指定城市 —— 先追问城市名，不要猜测
       - 用户问的是气候/四季特征（如"北京夏天热吗"）—— 可以直接回答常识
       - 参数 city 只传城市名，不要加"天气""今天"等后缀

    注意: 仅支持城市名，不支持省/国家/区域名。如果查询失败，告诉用户检查城市名拼写。
    """
    # 基本校验：防止空字符串或明显非城市名输入
    city = city.strip()
    if not city:
        return (
            "❌ 城市名不能为空。请提供具体的城市名称，如 '北京'、'Shanghai'、'Tokyo'。"
        )
    if len(city) > 50 or any(ch in city for ch in ("天气", "今天", "?", "吗")):
        return (
            f"❌ 输入 '{city}' 看起来不是有效的城市名。请只传城市名称（中文或英文均可），"
            f"不要附加其他文字。例如: '北京'、'Shanghai'。"
        )

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"https://wttr.in/{city}",
                params={"format": "j1"},
                headers={"Accept-Language": "zh-CN"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        return (
            f"❌ 天气查询网络失败: {exc}\n"
            f"   请稍后重试。如果持续失败，可能是城市名 '{city}' 不在支持范围内。"
        )
    except Exception as exc:
        return f"❌ 天气查询失败: {exc}"

    try:
        current = data.get("current_condition", [{}])[0]
        weather_info = data.get("weather", [])

        def _desc(entry: dict) -> str:
            desc_list = entry.get("weatherDesc", [])
            return desc_list[0].get("value", "?") if desc_list else "?"

        def _lang_desc(entry: dict) -> str:
            zh_list = entry.get("lang_zh", [])
            return zh_list[0].get("value", "?") if zh_list else _desc(entry)

        lines = [
            f"📍 城市: {city}",
            "",
            "🌤 当前天气:",
            f"    温度: {current.get('temp_C', '?')}°C "
            f"(体感 {current.get('FeelsLikeC', '?')}°C)",
            f"    天气: {_lang_desc(current)}",
            f"    湿度: {current.get('humidity', '?')}%",
            f"    风力: {current.get('winddir16Point', '?')} "
            f"{current.get('windspeedKmph', '?')} km/h",
            f"    能见度: {current.get('visibility', '?')} km",
            f"    紫外线指数: {current.get('uvIndex', '?')}",
            "",
        ]

        if weather_info:
            lines.append("📅 未来几天预报:")
            for day in weather_info[:3]:
                date_str = day.get("date", "?")
                max_c = day.get("maxtempC", "?")
                min_c = day.get("mintempC", "?")
                avg_c = day.get("avgtempC", "?")
                hourly = day.get("hourly", [])
                desc = "?"
                if hourly:
                    midday = hourly[min(len(hourly) // 2, len(hourly) - 1)]
                    desc = _lang_desc(midday)
                lines.append(
                    f"  {date_str}: {desc}, {min_c}°C ~ {max_c}°C (平均 {avg_c}°C)"
                )
            lines.append("")

        lines.append("💡 数据来源: wttr.in (免费天气服务)")
        return "\n".join(lines)

    except Exception as exc:
        return (
            f"❌ 解析天气数据失败: {exc}\n"
            f"   可能是城市 '{city}' 返回了非预期格式。请尝试用英文名。"
        )


@tool
def web_scraper(
    url: Annotated[
        str,
        Field(description="要抓取的网页 URL，必须以 http:// 或 https:// 开头"),
    ],
) -> str:
    """抓取指定网页，提取标题、描述和正文文本内容。

    底层使用 httpx 获取 HTML，再经由 BeautifulSoup 解析提取纯文本。

    ✅ 使用场景：
       - 用户给了具体 URL 并说"帮我看看这个网页写了什么""抓取这个链接的内容"
       - 用户需要了解某篇文章或网页的主要内容
       - 用户问"这个页面讲的什么"

    ❌ 不要使用：
       - 用户只是提到网站名但没有给具体 URL —— 直接回答常识
       - URL 是视频、PDF、图片等非 HTML 资源 —— 告诉用户当前不支持
       - 用户要求登录后才能访问的页面 —— 无法抓取
       - 用户要"搜索"某个关键词但没有给 URL —— 直接告知需要具体链接

    限制: 仅抓取 HTTP/HTTPS 页面，超时 15 秒，正文输出截断至 3000 字符。
    """
    from urllib.parse import urlparse

    from bs4 import BeautifulSoup

    # ---- 基本安全校验 ----
    url = url.strip()
    if not url:
        return "❌ URL 不能为空。请提供完整的网页链接，例如 'https://example.com/article'。"

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return (
            f"❌ 不支持的协议: '{parsed.scheme}'。仅支持 http 和 https。\n"
            f"   请提供以 http:// 或 https:// 开头的完整 URL。"
        )
    if not parsed.netloc:
        return f"❌ URL 缺少域名: '{url}'。请提供完整的网页链接。"

    # ---- 发起请求 ----
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; DemoAgent/1.0; "
                        "learning-purpose-only)"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
            resp.raise_for_status()
    except httpx.ConnectTimeout:
        return f"❌ 连接超时: 无法在 15 秒内连接到 {parsed.netloc}。请检查 URL 是否正确、网站是否可访问。"
    except httpx.ReadTimeout:
        return f"❌ 读取超时: 服务器响应过慢。请稍后重试。"
    except httpx.HTTPStatusError as exc:
        return (
            f"❌ HTTP 错误 {exc.response.status_code}: {url}\n"
            f"   可能是页面不存在 (404)、需要登录 (401/403) 或服务器错误 (5xx)。"
        )
    except httpx.HTTPError as exc:
        return f"❌ 网络请求失败: {exc}"

    # ---- 检查 content-type ----
    content_type = resp.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return (
            f"⚠️  非 HTML 内容: Content-Type 为 '{content_type}'。\n"
            f"   当前仅支持解析 HTML 网页。PDF、图片、视频等资源无法抓取。"
        )

    # ---- 解析 HTML ----
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        return f"❌ HTML 解析失败: {exc}"

    # 移除 script / style / nav / footer 等噪音标签
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # ---- 提取关键信息 ----
    title = _extract_title(soup, url)
    description = _extract_description(soup)

    # 提取正文（优先 <article>、<main>，回退到 <body>）
    main = soup.find("article") or soup.find("main") or soup.find("body")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # 清理：合并连续空行、截断
    import re

    text = re.sub(r"\n{3,}", "\n\n", text)
    max_chars = 3000
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (内容已截断，共 {len(text)} 字符，显示前 {max_chars} 字符)"

    # ---- 组装输出 ----
    lines = [
        f"🌐 网页抓取结果",
        f"",
        f"📌 标题: {title}",
    ]
    if description:
        lines.append(f"📝 描述: {description}")
    lines.extend([
        f"🔗 URL: {url}",
        f"",
        f"📄 正文内容:",
        f"{text}",
    ])
    return "\n".join(lines)


def _extract_title(soup: "BeautifulSoup", fallback_url: str) -> str:
    """提取网页标题：<title> → <h1> → og:title → URL fallback."""
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)[:200]
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()[:200]
    return fallback_url


def _extract_description(soup: "BeautifulSoup") -> str:
    """提取描述：meta description → og:description → 空."""
    for attr in ("name", "property"):
        for val in ("description", "og:description"):
            meta = soup.find("meta", {attr: val})
            if meta and meta.get("content"):
                desc = meta["content"].strip()
                return desc[:300] if len(desc) > 300 else desc
    return ""


# ===================================================================
# 4. 策略 ②：系统提示词 —— 路由规则 + Few-shot + 歧义消解
# ===================================================================

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具完成任务。

# 工具选择原则
根据用户消息中的关键词和意图，匹配对应工具。每个工具的 docstring 中已详细列出 ✅ 使用场景和 ❌ 不要使用的场景，请严格遵循。

# 关键歧义消解（最容易被混淆的场景）
- 带城市的天气查询 → get_weather。但"北京有多少人口""北京在哪个省"→ 直接回答常识，不调工具
- 明确的数学算式 → calculator。但"我今年25岁"这类陈述 → 不调工具
- 用户给了具体文本并说"统计字数" → word_count。但泛泛地问"怎么统计字数" → 直接解释方法
- "现在几点" → get_current_datetime。"今天天气" → get_weather。两者不要混淆
- 用户给了 URL 并要求查看内容 → web_scraper。但只提网站名没给 URL → 追问具体链接
- 用户只是感叹（如"今天好热啊"）→ 不调工具，友好回应

# 注意事项
- 每次只调用真正需要的工具，不要多调
- 工具参数按 docstring 中的示例格式填写，不要添加无关后缀
- 工具返回 ❌ 错误时，读错误信息，修正后重试一次
- 不确定是否需要工具时，直接回答，不强行调用
"""


# ===================================================================
# 5. 策略 ④：Middleware —— 拦截每次 tool call，记录日志 + 基础校验
# ===================================================================

class ToolMonitorMiddleware(AgentMiddleware):
    """工具调用监控中间件。

    职责:
    1. 记录每次工具调用的名称、参数、结果（可观察性）
    2. 对常见错误做预警日志（空参数、可疑参数等）

    这不是硬拦截 —— 即使参数可疑，仍然放行给工具执行，
    让工具自身的校验逻辑（策略⑤）来最终决定。
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        # ---- 执行前日志 ----
        logger.info("🔧 调用工具: %s | 参数: %s", tool_name, tool_args)

        # ---- 可选预警规则（不打断，只打日志） ----
        self._warn_if_suspicious(tool_name, tool_args)

        # ---- 执行工具 ----
        result = handler(request)

        # ---- 执行后日志 ----
        content_preview = (
            str(result.content)[:120] + "..."
            if len(str(result.content)) > 120
            else str(result.content)
        )
        status = getattr(result, "status", "?")
        logger.info("📤 工具返回 [%s]: %s", status, content_preview)

        return result

    # ------------------------------------------------------------------
    # 预警规则（可按业务需要扩展）
    # ------------------------------------------------------------------
    @staticmethod
    def _warn_if_suspicious(tool_name: str, args: dict) -> None:
        if tool_name == "calculator":
            expr = str(args.get("expression", ""))
            if not expr.strip():
                logger.warning("⚠️  calculator 收到空表达式！")
            elif len(expr) > 500:
                logger.warning("⚠️  calculator 表达式过长 (%d 字符)，可能传入了非表达式内容", len(expr))

        elif tool_name == "get_weather":
            city = str(args.get("city", ""))
            if not city.strip():
                logger.warning("⚠️  get_weather 收到空城市名！")
            elif any(kw in city for kw in ("天气", "今天", "?", "吗", "怎么样")):
                logger.warning(
                    "⚠️  get_weather 城市名含无关后缀: '%s'，可能是 LLM 未正确提取城市名", city
                )

        elif tool_name == "json_formatter":
            raw = str(args.get("json_string", ""))
            if not raw.strip():
                logger.warning("⚠️  json_formatter 收到空输入！")
            elif not raw.strip().startswith(("{", "[")):
                logger.warning("⚠️  json_formatter 输入可能不是 JSON: '%s...'", raw[:60])

        elif tool_name == "word_count":
            text = str(args.get("text", ""))
            if not text.strip():
                logger.warning("⚠️  word_count 收到空文本！")

        elif tool_name == "get_current_datetime":
            # 这个工具不需要参数，如果传了参数就是异常
            if args:
                logger.warning("⚠️  get_current_datetime 不需要参数，但收到了: %s", args)


# ===================================================================
# 6. 构建 Agent
# ===================================================================

def build_agent():
    """构建并返回带监控中间件的 agent。"""
    llm = _build_llm()
    tools = [
        calculator,
        get_current_datetime,
        word_count,
        json_formatter,
        get_weather,
        web_scraper,
    ]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[ToolMonitorMiddleware()],
    )


# ===================================================================
# 7. 构建会话记忆
# ===================================================================

def build_memory() -> tuple[ConversationMemory | None, object | None, str]:
    """构建 Redis 会话记忆 + 可选 Gatekeeper 实例。

    如果 Redis 不可用，返回 None 并回退到无记忆模式。
    如果 GATEKEEPER_ENABLED=true，同时构建 MemoryGatekeeper。

    Returns:
        (memory, gatekeeper, session_id) — memory 为 None 表示降级为无记忆模式。
    """
    if not _MEMORY_AVAILABLE:
        print("⚠️  记忆模块未安装（app.core.memory），将以无记忆模式运行")
        return None, None, ""

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    ttl = int(os.getenv("REDIS_SESSION_TTL", "3600"))

    # 尝试连接 Redis
    try:
        store = RedisSessionStore(
            redis_url=redis_url,
            password=redis_password,
            default_ttl=ttl,
        )
        store.client.ping()
    except Exception as exc:
        print(f"⚠️  Redis 连接失败 ({exc})，将以无记忆模式运行")
        print(f"   启动 Redis: docker compose up -d redis")
        return None, None, ""

    memory = ConversationMemory(store=store)

    # ── 可选: 构建 Gatekeeper ──────────────────────────────────────
    gatekeeper = None
    if os.getenv("GATEKEEPER_ENABLED", "").lower() in ("1", "true", "yes"):
        if MemoryGatekeeper is not None:
            entry_ttl = int(os.getenv("REDIS_ENTRY_TTL", "2592000"))
            entry_store = RedisSessionStore(
                redis_url=redis_url,
                password=redis_password,
                default_ttl=entry_ttl,
                key_prefix="mem:entry",
            )
            # 复用 demo agent 的 LLM 实例作为分类器后端
            llm = _build_llm()
            classifier = MemoryClassifier(
                llm_func=lambda prompt: llm.invoke(prompt).content,
            )
            gatekeeper = MemoryGatekeeper(store=entry_store, classifier=classifier)
            print("🛡️  Gatekeeper 已启用（结构化记忆准入过滤）")
        else:
            print("⚠️  Gatekeeper 模块不可用，结构化记忆功能关闭")

    # 生成默认 session_id
    default_id = f"demo-{datetime.now().strftime('%Y%m%d')}"
    custom = input(f"📝 Session ID (回车用默认 '{default_id}'): ").strip()
    session_id = custom if custom else default_id

    # 检查是否为已有会话
    if memory.session_exists(session_id):
        msg_count = memory.get_message_count(session_id)
        print(f"📂 恢复已有会话 '{session_id}'（{msg_count} 条历史消息）")
    else:
        print(f"🆕 新建会话 '{session_id}'")

    return memory, gatekeeper, session_id


# ===================================================================
# 7b. 工具调用信息提取
# ===================================================================

def _extract_tool_info_from_messages(messages: list) -> TurnToolInfo | None:
    """从 agent 返回的消息列表中提取本轮工具调用摘要。

    用于 Gatekeeper.process_turn() 的 tool_info 参数。
    """
    if TurnToolInfo is None:
        return None

    tool_msgs = [m for m in messages if hasattr(m, "type") and m.type == "tool"]
    ai_with_calls = [m for m in messages if hasattr(m, "tool_calls") and m.tool_calls]

    if not ai_with_calls and not tool_msgs:
        return TurnToolInfo()  # 无工具调用

    # 收集所有工具调用
    tool_names: list[str] = []
    all_args: dict[str, Any] = {}
    success = True
    errors: list[str] = []
    result_parts: list[str] = []

    for ai_msg in ai_with_calls:
        for tc in ai_msg.tool_calls:
            tool_names.append(tc.get("name", "unknown"))
            all_args.update(tc.get("args", {}))

    for tm in tool_msgs:
        content = str(tm.content) if hasattr(tm, "content") else ""
        if content.startswith("❌") or "错误" in content or "失败" in content:
            success = False
            errors.append(content[:200])
        result_parts.append(content[:200])

    return TurnToolInfo(
        tool_name=", ".join(tool_names) if tool_names else "",
        args=all_args,
        result_preview=" | ".join(result_parts)[:500],
        success=success,
        error_message="; ".join(errors) if errors else None,
    )


# ===================================================================
# 8. 交互式运行
# ===================================================================

def _print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def main() -> None:
    """交互式 Agent 对话循环，集成 Redis 会话记忆 + Gatekeeper 结构化记忆。

    每次对话自动加载历史消息并追加当前输入，agent 可在多轮对话中保持上下文。
    Gatekeeper 在每轮对话后自动提取结构化记忆条目（需 GATEKEEPER_ENABLED=true）。

    特殊命令:
        /clear   — 清除当前会话记忆（含 Gatekeeper 条目）
        /memory  — 查看原始消息窗口状态
        /entries — 查看 Gatekeeper 结构化记忆条目
        /help    — 显示帮助
    """
    _print_separator()
    print("🤖 简易 LangChain Agent 示例（含工具调用监控 + Redis 会话记忆）")
    print(f"   模型: {os.getenv('LLM_MODEL_NAME', 'gpt-4o-mini')}")
    print("   可用工具: calculator, get_current_datetime, word_count, json_formatter, get_weather, web_scraper")
    _print_separator()

    agent = build_agent()
    memory, gatekeeper, session_id = build_memory()

    if memory:
        mode_parts = ["Redis 会话记忆（多轮上下文保持）"]
        if gatekeeper:
            mode_parts.append("+ Gatekeeper（结构化记忆准入）")
        print(f"🧠 记忆模式: {' '.join(mode_parts)}")
    else:
        print("🧠 记忆模式: 无记忆（每轮独立）")
    print("输入问题开始对话，/help 查看命令，quit / exit / q 退出")
    _print_separator()

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # ---- 退出命令 ----
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        # ---- /clear: 清除会话记忆 ----
        if user_input.strip() == "/clear":
            if memory:
                memory.clear(session_id)
                # Gatekeeper 条目：逐个标记为 expired
                if gatekeeper:
                    try:
                        entries = gatekeeper.list_entries(session_id)
                        for entry in entries:
                            gatekeeper.delete_entry(entry.entry_id)
                        print(f"🧹 会话记忆已清除（含 {len(entries)} 条 Gatekeeper 条目），开始全新对话。")
                    except Exception:
                        print("🧹 会话记忆已清除，开始全新对话。")
                else:
                    print("🧹 会话记忆已清除，开始全新对话。")
            else:
                print("⚠️  无记忆模式，无需清除。")
            continue

        # ---- /memory: 查看记忆状态 ----
        if user_input.strip() == "/memory":
            if memory and memory.session_exists(session_id):
                meta = memory.load_metadata(session_id)
                msg_count = memory.get_message_count(session_id)
                print(f"📊 会话记忆状态:")
                print(f"   Session ID:    {session_id}")
                print(f"   消息总数:      {msg_count}")
                print(f"   总轮数:        {meta.get('total_turns', '?')}")
                print(f"   存储消息数:     {meta.get('stored_messages', '?')}")
                print(f"   可信度:        {meta.get('confidence', '?')}")
                print(f"   创建时间:      {meta.get('created_at', '?')}")
                print(f"   最后更新:      {meta.get('updated_at', '?')}")
                print(f"   状态:          {meta.get('status', '?')}")
                print(f"   数据版本:      {meta.get('version', '?')}")
                ttl = int(os.getenv("REDIS_SESSION_TTL", "3600"))
                print(f"   TTL 设置:      {ttl}s ({ttl // 60} 分钟)")
            elif memory:
                print(f"📊 会话 '{session_id}' 暂无记忆数据。")
            else:
                print("⚠️  无记忆模式，无会话状态。")
            continue

        # ---- /entries: 查看 Gatekeeper 条目 ----
        if user_input.strip() == "/entries":
            if gatekeeper:
                entries = gatekeeper.list_entries(session_id)
                if entries:
                    print(f"📋 结构化记忆条目 ({len(entries)} 条):")
                    for i, entry in enumerate(entries, 1):
                        status_icon = {
                            "active": "✅", "pending_review": "⏳",
                            "conflicted": "⚠️", "archived": "📦", "expired": "💤",
                        }.get(entry.status.value, "❓")
                        print(f"  {i}. {status_icon} [{entry.entry_type.value}] "
                              f"{entry.summary or entry.content[:80]} "
                              f"(conf={entry.confidence:.0%}, v{entry.version})")
                else:
                    print(f"📋 会话 '{session_id}' 暂无结构化记忆条目。")
            elif memory:
                print("⚠️  Gatekeeper 未启用。设置 GATEKEEPER_ENABLED=true 以启用。")
            else:
                print("⚠️  无记忆模式，无结构化条目。")
            continue

        # ---- /help: 显示帮助 ----
        if user_input.strip() == "/help":
            print("可用命令:")
            print("  /clear   — 清除当前会话记忆，开始全新对话")
            print("  /memory  — 查看原始消息窗口状态（消息数、轮数等）")
            print("  /entries — 查看 Gatekeeper 结构化记忆条目")
            print("  /help    — 显示此帮助")
            print("  quit / exit / q  — 退出")
            print()
            print("直接输入问题即可开始对话。agent 会记住本轮对话上下文。")
            print("设置 GATEKEEPER_ENABLED=true 可启用结构化记忆准入过滤。")
            continue

        # ---- 正常对话流程 ----
        print("🤖 Agent 思考中...")
        _print_separator()

        try:
            # 加载历史消息（有记忆模式）或使用空列表（无记忆模式）
            if memory:
                history = memory.load_messages(session_id)
            else:
                history = []

            # 追加当前用户消息
            messages = history + [HumanMessage(content=user_input)]

            # 调用 agent
            result = agent.invoke({"messages": messages})

            # 提取本次对话产生的完整消息列表并持久化
            all_messages = result.get("messages", [])
            if memory:
                try:
                    memory.save_messages(session_id, all_messages)
                except Exception as exc:
                    logger.warning("⚠️  保存会话记忆失败: %s", exc)

        except Exception as exc:
            print(f"❌ Agent 调用失败: {exc}")
            continue

        # 提取最终回答
        final_answer = None
        for msg in reversed(all_messages):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                final_answer = msg.content
                break

        # Gatekeeper: 结构化记忆提取（在 final_answer 提取后执行）
        if gatekeeper and final_answer:
            try:
                tool_info = _extract_tool_info_from_messages(all_messages)
                gatekeeper.process_turn(
                    session_id=session_id,
                    turn_index=len(all_messages) // 2,
                    user_message=user_input,
                    assistant_message=final_answer,
                    tool_info=tool_info,
                )
            except Exception:
                logger.warning("⚠️  Gatekeeper 提取失败", exc_info=True)

        if final_answer:
            print(f"🤖 Agent: {final_answer}")
        else:
            print("🤖 Agent: (无输出)")

        # 汇总工具调用
        tool_msgs = [
            msg for msg in all_messages
            if hasattr(msg, "type") and msg.type == "tool"
        ]
        ai_with_tool_calls = [
            msg for msg in all_messages
            if hasattr(msg, "tool_calls") and msg.tool_calls
        ]
        if ai_with_tool_calls:
            print(f"\n📊 本轮工具调用: {len(ai_with_tool_calls)} 次 LLM 决策 → {len(tool_msgs)} 次实际执行")

        # 显示记忆状态
        status_parts: list[str] = []
        if memory and memory.session_exists(session_id):
            msg_count = memory.get_message_count(session_id)
            status_parts.append(f"消息: {msg_count}")
        if gatekeeper:
            try:
                entries = gatekeeper.list_entries(session_id)
                if entries:
                    status_parts.append(f"条目: {len(entries)}")
            except Exception:
                pass
        if status_parts:
            print(f"🧠 会话记忆 | {' | '.join(status_parts)} | 会话: {session_id}")


if __name__ == "__main__":
    main()
