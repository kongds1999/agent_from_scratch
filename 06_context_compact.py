#!/usr/bin/env python3
"""
s06_context_compact.py: 自动化 3 层上下文管理的实现。

Motto: "上下文会充满；你需要一种方法来制造空间"

该模块解决了 LLMs 的 '上下文窗口' 限制。随着对话的发展，
逐字历史会消耗更多的 token，最终导致 API 错误或性能下降。
该脚本实现了一种滚动压缩机制，类似于 'Claude Code' CLI。

压缩架构:
    1. 层 1 (逐字保留): 最后 N 条消息会被精确保留，
       以维护立即的 "短期" 上下文。
    2. 层 2 (摘要): 较早的消息会被发送到模型进行压缩，
       生成一个简洁的决策和操作总结。
    3. 层 3 (持久化): 生成的摘要会被写入 Markdown 文件
       (`.agent_memory.md`) 中，允许上下文在不同会话或重启之间持续存在。

触发:
    当对话历史的估计字符数超过预定义阈值时，会自动触发压缩。

压缩过程:
    1. Layer 1 (逐字保留): 最后 N 条消息会被精确保留，
       以维护立即的 "短期" 上下文。
    2. Layer 2 (摘要): 较早的消息会被发送到模型进行压缩，
       生成一个简洁的决策和操作总结。
    3. Layer 3 (持久化): 生成的摘要会被写入 Markdown 文件
       (`.agent_memory.md`) 中，允许上下文在不同会话或重启之间持续存在。

触发:
    当对话历史的估计字符数超过预定义阈值时，会自动触发压缩。
"""

# === Standard Library Imports ===
import os      # Operating system interfaces
import sys     # System-specific parameters and functions
from pathlib import Path  # Object-oriented filesystem paths
from typing import List, Dict, Any, Union, Optional  # For strict type hinting

# === Local Module Imports ===

from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, RunConfig, MultiProvider, set_tracing_disabled, set_default_openai_api
from agents.extensions.models.litellm_model import LitellmModel
from tools import AGENT_TOOLS, discover_skills, SKILLS_DIR, SKILL_TOOLS

load_dotenv()
set_tracing_disabled(True)  # Disable tracing for cleaner output
set_default_openai_api("chat_completions")  # 兼容 LiteLLM / 第三方中转

import json

# === Configuration and Constants ===

MODEL: str = "gpt-6-astra"

# Custom model configuration for LiteLLM proxy
custom_model = LitellmModel(
    model=MODEL,
    base_url=os.getenv("LITELLM_PROXY_API_BASE", "")+"/v1/responses",
    api_key=os.getenv("LITELLM_PROXY_API_KEY", "")
)

# 是否开启 DEBUG 模式（默认可由环境变量 DEBUG 控制，也可在终端输入 /debug 随时切换）
DEBUG: bool = os.getenv("DEBUG", "1") in {"1", "true", "True"}

# 1. 启动时发现技能，把轻量索引写进系统提示（对应原版的 discovery 阶段）
_initial_skills = discover_skills()
_skill_index_str = "\n".join(
    f"  - {n}: {d}" for n, d in _initial_skills.items()
) or "  (none currently installed)"


_base_url = os.environ.get("OPENAI_BASE_URL", "")
if not _base_url.endswith("/v1"):
    _base_url = f"{_base_url}/v1"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=_base_url,
    max_retries=3,
    timeout=60,
)

# 触发压缩的阈值 (approx. 4k chars ≈ 1k tokens)
COMPRESS_THRESHOLD: int = 2000

# 完整保留的最近消息数量 (Layer 1)
KEEP_RECENT: int = 6

# 长期记忆存在路径 (Layer 3)
MEMORY_FILE: Path = Path(".agent_memory.md")

# === Debug Utility Functions ===

def print_full_context(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    round_num: int = 1,
    title: str = "完整上下文 (Full Context)"
) -> None:
    """
    打印当前轮次的完整对话上下文（包含 System Prompt、所有历史轮次、工具调用等全量数据）。
    """
    full_list: List[Dict[str, Any]] = []
    if system_prompt:
        full_list.append({
            "role": "system",
            "content": system_prompt
        })
    full_list.extend(messages)

    total_chars = _estimate_size(messages) + (len(system_prompt) if system_prompt else 0)

    print(f"\n\033[36m{'=' * 22} [Round {round_num}] {title} {'=' * 22}\033[0m")
    print(f"\033[90m[概览] 累计条目数: {len(full_list)} 条 | 总估算字符: {total_chars} / {COMPRESS_THRESHOLD} chars\033[0m\n")
    print(json.dumps(full_list, indent=2, ensure_ascii=False))
    print(f"\033[36m{'=' * 72}\033[0m\n")

# === Context Utility Functions ===

def _extract_message_text(m: Dict[str, Any]) -> tuple[str, str]:
    """
    从对话历史条目中安全提取 (角色/类型, 文本内容)。
    兼容 Agents SDK 的各种消息格式（user, assistant, tool_call, tool_output, reasoning）。
    """
    m_type = m.get("type")
    role = m.get("role") or m_type or "system"

    if m_type == "function_call":
        name = m.get("name", "function")
        args = m.get("arguments", "")
        return f"tool_call:{name}", str(args)
    elif m_type == "function_call_output":
        return "tool_output", str(m.get("output", ""))
    elif m_type == "reasoning":
        summaries = m.get("summary", [])
        text = " ".join(
            s.get("text", "") if isinstance(s, dict) else str(s)
            for s in summaries
        )
        return "reasoning", text

    content = m.get("content", "")
    if isinstance(content, str):
        return str(role), content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(str(block.get("text", "") or block.get("content", "")))
            elif hasattr(block, "text"):
                text_parts.append(str(block.text or ""))
            else:
                text_parts.append(str(block))
        return str(role), " ".join(text_parts)
    return str(role), str(content or "")


def _estimate_size(messages: List[Dict[str, Any]]) -> int:
    """
    估计对话历史的总字符数。
    这是一个代理，用于确定何时压缩上下文。
    """
    total = 0
    for msg in messages:
        _, text = _extract_message_text(msg)
        total += len(text)
    return total


def _summarize(messages: List[Dict[str, Any]]) -> str:
    """
    使用 LLM 来压缩对话历史的一个片段。
    """
    lines = []
    for m in messages:
        role_label, text = _extract_message_text(m)
        if text.strip():
            lines.append(f"[{role_label}]: {text}")
    text_to_summarize = "\n\n".join(lines)

    if DEBUG:
        print("\n\033[36m" + "=" * 18 + " [DEBUG] CLIENT (SUMMARIZE) REQUEST " + "=" * 18 + "\033[0m")
        print(f"Model: {MODEL}")
        print(f"Chars to summarize: {len(text_to_summarize)}")
        preview = text_to_summarize[:1000] + ("\n... [truncated]" if len(text_to_summarize) > 1000 else "")
        print(preview)
        print("\033[36m" + "=" * 70 + "\033[0m\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a context compressor. Summarize the provided conversation history "
                "concisely. Retain all critical technical decisions, file paths mentioned, "
                "code changes made, and pending tasks. Ignore trivial back-and-forth."
            )},
            {"role": "user", "content": f"Summarize this history:\n\n{text_to_summarize[:20000]}"}
        ],
        max_tokens=2000,
    )

    summary_text = response.choices[0].message.content.strip()
    if DEBUG:
        print("\n\033[36m" + "=" * 18 + " [DEBUG] CLIENT (SUMMARIZE) RESPONSE " + "=" * 17 + "\033[0m")
        print(summary_text)
        print("\033[36m" + "=" * 70 + "\033[0m\n")

    return summary_text


def maybe_compress(messages: List[Dict[str, Any]]) -> bool:
    """
    评估上下文大小并在超过阈值时执行压缩。
    此函数会修改 'messages' 列表，将旧的对话轮次替换为一个总结轮次。
    """
    if _estimate_size(messages) < COMPRESS_THRESHOLD:
        return False

    if len(messages) <= KEEP_RECENT:
        return False

    # 寻找以 user 消息开头的分割点，避免破坏 tool_call / tool_output 的配对完整性
    split_idx = max(0, len(messages) - KEEP_RECENT)
    while split_idx < len(messages) and messages[split_idx].get("role") != "user":
        split_idx += 1

    if split_idx >= len(messages):
        split_idx = max(0, len(messages) - KEEP_RECENT)
        while split_idx > 0 and messages[split_idx].get("role") != "user":
            split_idx -= 1

    if split_idx <= 0 or split_idx >= len(messages):
        return False

    print("\033[90m  [compress] Context large — condensing older history...\033[0m")

    old_messages = messages[:split_idx]
    recent_messages = messages[split_idx:]

    summary = _summarize(old_messages)

    try:
        MEMORY_FILE.write_text(
            f"# Agent Context Memory\n*Last updated: {os.getcwd()}*\n\n{summary}\n",
            encoding="utf-8",
        )
    except Exception as e:
        print(f"\033[31m  [error] Failed to persist memory: {e}\033[0m")

    messages.clear()

    # 1. 注入摘要作为新的起始上下文
    messages.append({
        "role": "user",
        "content": f"[Context summary of previous turns]:\n\n{summary}",
    })
    # 2. 添加 assistant 确认回复，维持 user/assistant 轮流交替规则
    messages.append({
        "role": "assistant",
        "content": "Understood. I have integrated the summary of our previous progress into my current context.",
    })
    # 3. 恢复最近的逐字消息
    messages.extend(recent_messages)

    print(f"\033[90m  [compress] Done. Collapsed {len(old_messages)} messages into 1 summary. Saved to {MEMORY_FILE}\033[0m")
    return True


# === Main Execution Block ===

def main() -> None:
    """
    Initializes the terminal interaction for the s06 'Context Compacting' agent.
    """
    # Initialize history
    history: List = []

    # 从之前会话中恢复上下文 (Session Persistence)
    if MEMORY_FILE.exists():
        try:
            mem_content = MEMORY_FILE.read_text(encoding="utf-8")
            print(f"\033[90m  [memory] Restoring context from {MEMORY_FILE}...\033[0m")
            # Seed the history with the saved memory
            history = [
                {"role": "user",      "content": f"[Previous Session Memory]:\n\n{mem_content}"},
                {"role": "assistant", "content": "Memory loaded. I am ready to continue where we left off."},
            ]
        except Exception as e:
            print(f"\033[31m  [error] Could not read memory file: {e}\033[0m")

    global DEBUG

    # UI Header
    print(f"\033[90ms06: context compression at ~{COMPRESS_THRESHOLD//1000}k chars | memory → {MEMORY_FILE}\033[0m")
    debug_status = "\033[32mON\033[0m" if DEBUG else "\033[31mOFF\033[0m"
    print(f"\033[90mdebug mode: {debug_status} \033[90m(type /debug to toggle, /context to view)\033[0m\n")

    print("\033[90m openai-agents-code v0.1.0 \033[0m")
    print(f"\033[90m skills dir: {SKILLS_DIR}\033[0m")

    agent = Agent(
        name="AgentAssistant",
        model=custom_model,
        instructions=f"You are a coding agent at {os.getcwd()}. Use tools to solve tasks. \n\nAvailable Skills Index:\n{_skill_index_str}",
        tools=[
            *AGENT_TOOLS,
            *SKILL_TOOLS
        ],
    )

    round_num = 1

    while True:
        try:
            query = input("\033[32m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[31mExiting...\033[0m")
            sys.exit(0)

        if not query or query.lower() in {"q", "exit", "quit"}:
            print("\033[31mExiting the REPL.\033[0m")
            break

        # 快捷切换 DEBUG 模式
        if query.lower() == "/debug":
            DEBUG = not DEBUG
            status = "\033[32mON\033[0m" if DEBUG else "\033[31mOFF\033[0m"
            print(f"\033[90m[debug] DEBUG mode toggled to {status}\033[0m\n")
            continue

        # 随时查看当前完整上下文
        if query.lower() in {"/context", "/ctx"}:
            print_full_context(history, system_prompt=agent.instructions, round_num=round_num, title="当前会话完整上下文")
            continue

        # 将用户新输入并入上下文
        current_input = history + [{"role": "user", "content": query}]

        # Runner 内部会自动处理多轮 Tool Call 循环（包括 load_skill 的惰性加载）
        # result = Runner.run_sync(agent, input=current_input, run_config=RUN_CONFIG)
        result = Runner.run_sync(agent, input=current_input)

        # 更新多轮对话历史
        history = result.to_input_list()

        # DEBUG: 打印本轮执行后的完整上下文（包含 System Prompt、历史轮次、本轮推理/工具/回答）
        if DEBUG:
            print_full_context(
                history, 
                system_prompt=agent.instructions, 
                round_num=round_num, 
                title="本轮交互后完整上下文 (Full Conversation Context)"
            )

        # 评估并在超限时压缩对话上下文
        compressed = maybe_compress(history)
        if compressed and DEBUG:
            print_full_context(
                history, 
                system_prompt=agent.instructions, 
                round_num=round_num, 
                title="压缩后新完整上下文 (Post-Compression Full Context)"
            )

        print("\n\033[32mFinal Answer:\033[0m")
        print(result.final_output)
        print()

        round_num += 1


if __name__ == "__main__":
    # Script entry point
    main()
