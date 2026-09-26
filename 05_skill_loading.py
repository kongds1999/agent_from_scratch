#!/usr/bin/env python3
"""
05_skill_loading.py: 按需加载 skill（OpenAI Agents SDK 版）。

Motto: "Load knowledge when you need it, not upfront"

该模块介绍一种名为 'Meta-Tooling' 的技术，用于解决"上下文窗口膨胀"问题。
不再将每种可能的指令、指南或特定 SOP 都包含在系统提示中，
而是允许 agent 去发现并加载特定 skill。

关键架构概念:
    1. 发现：agent 被赋予了一个轻量级的技能索引（名称和1行描述），并包含在其系统提示中。
    2. 惰性加载：只有当 agent 显式调用 `load_skill` 时，才会将技能的完整文档注入到对话中。
    3. 上下文效率：这允许 agent 访问数百个专业技能，而不会超过 token 限制或使模型混淆无关数据。

原版（手写版）把 list_skills / load_skill 写成 tool schema + dispatch 字典，再交给自制的
stream_loop 执行。换成 OpenAI Agents SDK 后，这两块都由 SDK 原生承担：
    - `@function_tool` 直接把 Python 函数变成工具，参数 schema 从类型注解自动生成。
    - `Runner` 自动处理多轮 tool call 循环，不再需要 stream_loop / EXTENDED_DISPATCH。

Skill Structure:
    Skills are stored in: skills/<skill_name>/SKILL.md
"""

# === Standard Library Imports ===
import os      # Operating system interfaces
import sys     # System-specific parameters and functions

from dotenv import load_dotenv
from agents import Agent, Runner, set_tracing_disabled, set_default_openai_api

# === Local Module Imports ===
# 基础文件/命令工具 + 两个 "meta-tooling" 工具（定义在 tools.py）
from tools import AGENT_TOOLS, discover_skills, SKILLS_DIR, SKILL_TOOLS

load_dotenv()
set_tracing_disabled(True)  # Disable tracing for cleaner output
set_default_openai_api("chat_completions")  # 兼容 LiteLLM / 第三方中转


# === Dynamic System Prompt Construction ===

# 1. 启动时发现技能，把轻量索引写进系统提示（对应原版的 discovery 阶段）
_initial_skills = discover_skills()
_skill_index_str = "\n".join(
    f"  - {n}: {d}" for n, d in _initial_skills.items()
) or "  (none currently installed)"

# 2. 构造 persona prompt，明确要求模型用 load_skill 惰性加载，而不是凭空猜测
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}.\n"
    "You have access to specialized 'Skills' (domain knowledge files). "
    "When a task requires specific knowledge (e.g., a specific framework, "
    "API, or language), call load_skill(name) to get full instructions. "
    "Do NOT guess or hallucinate details if a skill is available. "
    "Use list_skills whenever you need to refresh the index.\n\n"
    f"Available Skills Index:\n{_skill_index_str}"
)
print("\033[90mSYSTEM PROMPT:\033[0m")
print(SYSTEM)

# === Main Execution Block ===

def main() -> None:
    print("\033[90ms05: on-demand skill loading | list_skills · load_skill\033[0m\n")
    print("\033[90m openai-agents-code v0.1.0 \033[0m")
    print(f"\033[90m skills dir: {SKILLS_DIR}\033[0m")

    agent = Agent(
        name="SkillAssistant",
        model="openai/gpt-5.5",
        instructions=SYSTEM,
        # 基础工具 + 技能工具；Runner 会自动把它们暴露给模型
        tools=[
            *AGENT_TOOLS,
            *SKILL_TOOLS
        ],
    )

    # 存储多轮会话历史
    conversation_history = []

    while True:
        try:
            query = input("\033[32m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[31mExiting...\033[0m")
            sys.exit(0)

        if not query or query.lower() in {"q", "exit", "quit"}:
            print("\033[31mExiting the REPL.\033[0m")
            break

        # 将用户新输入并入上下文
        current_input = conversation_history + [{"role": "user", "content": query}]

        # Runner 内部会自动处理多轮 Tool Call 循环（包括 load_skill 的惰性加载）
        result = Runner.run_sync(agent, input=current_input)

        # 更新多轮对话历史
        conversation_history = result.to_input_list()

        print("\n\033[32mFinal Answer:\033[0m")
        print(result.final_output)
        print()


if __name__ == "__main__":
    main()
