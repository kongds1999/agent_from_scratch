#!/usr/bin/env python3
"""
s04_subagent.py: 用 OpenAI Agents SDK 原生 Handoffs 实现子 Agent 委托

Motto: "Break big tasks down; each subtask gets a clean context"
Motto: "拆解大的任务，每个子任务都有一个干净的上下文"

原版（Anthropic 手写版）用自制的 `spawn_subagent` 工具，在一个 tool call 里同步跑一个
全新的 agent loop，只把最终文本结论返回给 Lead。OpenAI Agents SDK 把这种「多 Agent 协作」
抽象成了两个原生能力，不需要我们再手写循环：

    1. Handoffs（任务转移 / 本文件重点）
       `Agent(..., handoffs=[...])`，SDK 自动把每个目标 agent 暴露成一个工具，
       名为 `transfer_to_<agent_name>`。调用后，对话由目标 agent「接管」。
       - 默认接收方能看到完整对话历史（这点和原版「干净上下文」不同）。
       - 可以用 `handoff(input_filter=...)` 过滤历史，用 `input_type` 让模型带上
         结构化参数，用 `on_handoff` 做副作用（打印日志、写库等）。

    2. Agent.as_tool()（原版 spawn_subagent 的严格等价物，见文件末尾注释）
       把子 agent 当作普通工具调用：子 agent 拿到的是「全新上下文」，且它只把结果
       返回给父 agent，不接管对话。需要「派活 + 汇总」时用它。

本文件演示第 1 种（Handoffs），把原版里所有手写逻辑都交给 SDK 处理。

State Mapping:
    - 原版:     Lead --spawn_subagent--> SubAgent(新历史) --return text--> Lead
    - Handoffs: Lead --transfer_to_x--> SubAgent(接管对话) --> Final Answer
"""

# === Standard Library Imports ===
import os      # Operating system interfaces (for pathing and environment)
import sys     # System-specific parameters (for exiting)
import asyncio # Async runtime for the streaming REPL

from dotenv import load_dotenv                                    # Load .env
from pydantic import BaseModel, Field                             # Handoff 的结构化输入

# === OpenAI Agents SDK Imports ===
from agents import (
    Agent,                        # Agent 定义
    Runner,                       # 驱动 agent loop（内部自动处理 tool calls / handoffs）
    handoff,                      # 自定义任务转移
    set_tracing_disabled,         # 关闭 tracing，输出更干净
    set_default_openai_api,       # 兼容第三方中转（chat completions）
)
from agents.exceptions import MaxTurnsExceeded                    # 超过 max_turns
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.handoffs import HandoffInputData                      # 自定义 input_filter
from agents.items import ToolCallItem, ToolCallOutputItem, ReasoningItem
from openai.types.responses import ResponseTextDeltaEvent         # 流式文本增量事件

# === Local Module Imports ===
from tools import AGENT_TOOLS  # 标准工具集：bash / read / write / grep / glob / revert

# 加载环境变量、关闭 tracing、强制走 chat_completions 以兼容第三方中转
load_dotenv()
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

# === Configuration and Constants ===

MODEL: str = "deepseek-v4-flash-vision-exp"


# === 子 Agent（Specialist）定义 ===
# 每个子 agent 都有自己的 instructions 和工具集；`handoff_description` 会追加到
# SDK 自动生成的 transfer 工具描述里，用来提示 Lead「什么时候该转移」。

explorer_agent = Agent(
    name="Explorer",
    model=MODEL,
    handoff_description=(
        "适合探索型子任务：通读代码库、搜索文件、理解实现细节。"
        "会产生大量中间输出，交给它可以把这些噪声隔离在 Lead 之外。"
    ),
    instructions=(
        f"You are an exploration subagent at {os.getcwd()}. "
        "Investigate the codebase thoroughly using read/grep/glob. "
        "When done, give a concise, structured summary of what you found."
    ),
    tools=AGENT_TOOLS,
)

coder_agent = Agent(
    name="Coder",
    model=MODEL,
    handoff_description=(
        "适合动手改代码的子任务：创建/修改文件、执行命令。"
        "高风险或有副作用的操作交给他。"
    ),
    instructions=(
        f"You are an implementation subagent at {os.getcwd()}. "
        "Make the requested code changes carefully, then summarize exactly what you changed."
    ),
    tools=AGENT_TOOLS,
)

reviewer_agent = Agent(
    name="Reviewer",
    model=MODEL,
    handoff_description=( # SDK 会把这个描述追加到 transfer_to_reviewer 工具的说明里，提示主agent什么时候该转移
        "适合验证型子任务：审查改动、运行测试、检查边界条件，最后给出结论。"
    ),
    instructions=(  # 子agent的instruction
        f"You are a review subagent at {os.getcwd()}. "
        "Review the work rigorously. Run tests or read code as needed. "
        "Report findings and a clear verdict at the end."
    ),
    tools=AGENT_TOOLS,
)

# 不使用原 agent 上下文，新开一个 agent 完成任务 & 返回完成任务的总结
explorer_tool = explorer_agent.as_tool(
    tool_name="spawn_explorer",
    tool_description="Spawn an explorer in a fresh context; it returns a summary.",
)
coder_tool = coder_agent.as_tool(
    tool_name="spawn_explorer",
    tool_description="Spawn a coder in a fresh context; it returns a summary.",
)
reviewer_tool = reviewer_agent.as_tool(
    tool_name="spawn_reviwer",
    tool_description="Spawn a reviewer in a fresh context; it returns a summary.",
)

# === Handoff 结构化输入 ===
# 文档：input_type 描述的是「任务转移工具调用的参数」，SDK 会校验模型生成的 JSON
# 并把解析后的对象传给 on_handoff。它不会替换接收方的主输入。
class Delegation(BaseModel):
    """模型在转移任务时给出的元数据，便于日志/审计。"""

    task: str = Field(description="交给子 agent 的具体子任务描述")


# 自定义 input_filter：清掉历史里的 tool call / tool output / reasoning 噪声，
# 让子 agent 的上下文不被上一轮的工具输出污染（对应原版「保护上下文」的动机）。
# 注意：SDK 自带的 agents.extensions.handoff_filters.remove_all_tools 更激进，
# 会连 handoff 自身的 item 一起删掉，导致流式 handoff 事件不再触发；
# 这里只过滤普通工具项，保留 handoff 可见性。
_TOOL_NOISE = (ToolCallItem, ToolCallOutputItem, ReasoningItem)
_NOISY_RAW_TYPES = {"function_call", "function_call_output", "reasoning"}


def strip_tool_noise(data: HandoffInputData) -> HandoffInputData:
    """去掉工具噪声，但保留 handoff 的 call/output item。"""

    def keep(items):
        return tuple(i for i in items if not isinstance(i, _TOOL_NOISE))

    history = data.input_history
    if isinstance(history, tuple):
        history = tuple(i for i in history if i.get("type") not in _NOISY_RAW_TYPES)

    return data.clone(
        input_history=history,
        pre_handoff_items=keep(data.pre_handoff_items),
        new_items=keep(data.new_items),
    )


def make_handoff(target: Agent):
    """为某个子 agent 创建一个带日志、结构化输入和上下文过滤的 Handoff。

    - on_handoff：转移发生时执行，打印洋红色日志（对应原版 run_subagent 的提示）。
    - input_type：让模型在 transfer 工具调用里带上 `task` 字段。
    - input_filter：过滤子 agent 看到的历史（见 strip_tool_noise）。
    """

    async def on_handoff(ctx, data: Delegation) -> None:
        # 洋红色（\033[35m），与原版子 agent 的输出颜色保持一致
        print(f"\033[35m  [subagent] delegating to {target.name}: {data.task[:80]}\033[0m")

    return handoff(
        target,
        on_handoff=on_handoff,
        input_type=Delegation,
        input_filter=strip_tool_noise,
    )


# === Lead Agent（Manager）===
# 使用 SDK 推荐的 prompt 前缀，告诉模型这是一个多 agent 系统，且转移是后台完成的。
lead_agent = Agent(
    name="Lead",
    model=MODEL,
    instructions=prompt_with_handoff_instructions(
        f"You are the lead coding agent at {os.getcwd()}. "
        "For complex, isolated, or noisy subtasks (exploration, risky edits, verification), "
        "delegate by transferring to the right specialist instead of doing it yourself. "
        "Do simple tasks directly."
    ),
    # 方式一：使用 as_tool() 提交任务说明给子agent，并期待返回任务总结
    tools=[*AGENT_TOOLS, explorer_tool, coder_tool, reviewer_tool],
    # 方式二：使用 handoffs 将当前上下文（已过滤）提交给子agent继续执行
    handoffs=[
        make_handoff(explorer_agent), # 相当于添加一个 transfer_to_explorer 工具，专门用来转移给 Explorer
        make_handoff(coder_agent),
        make_handoff(reviewer_agent),
    ],
)

# prompt_with_handoff_instructions() 会在模型的系统 prompt 里自动追加一段说明，告诉它「这是一个多 agent 系统，转移是后台完成的」。
# System context
# You are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.


# === Main Execution Block ===

async def main() -> None:
    print("\033[90ms04: subagent isolation | handoffs → specialist takes over\033[0m\n")
    print("\033[90m openai-agents-code v0.1.0 \033[0m")

    # 存储多轮会话历史（SDK 用 to_input_list() 生成，包含 handoff 产生的所有 item）
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

        current_input = conversation_history + [{"role": "user", "content": query}]

        # Runner 内部自动处理 tool call 循环与 handoff 转移
        result = Runner.run_streamed(lead_agent, input=current_input, max_turns=50)

        print("\n\033[32mFinal Answer:\033[0m")
        try:
            async for event in result.stream_events():
                # 1) 文本增量：实时打印最终回答
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    if event.data.delta:
                        print(event.data.delta, end="", flush=True)

                # 2) Handoff 事件：展示「谁把对话移交给了谁」
                elif event.type == "run_item_stream_event" and event.name == "handoff_occured":
                    item = event.item
                    print(
                        f"\n\033[35m  [handoff] {item.source_agent.name} -> "
                        f"{item.target_agent.name}\033[0m"
                    )

                # 3) Agent 切换事件：任务转移后由新 agent 接管
                elif event.type == "agent_updated_stream_event":
                    print(f"\033[35m  [agent] now running: {event.new_agent.name}\033[0m")
        except MaxTurnsExceeded as e:
            print(f"\n\033[31m[!] {e} — 增大 max_turns 或传 None\033[0m")

        # 更新多轮对话历史（handoff 也在同一次 run 内，历史会被完整保留）
        conversation_history = result.to_input_list()
        print(f"\n\033[90m[last agent] {result.last_agent.name}\033[0m")
        print()

# 两者区别（SDK 官方表述）：
#     - Handoff：接收方能看到对话历史，并「接管」对话，不会把控制权交回。
#     - as_tool：接收方只拿到传入的输入（无历史），以工具结果形式返回给父 agent。


if __name__ == "__main__":
    asyncio.run(main())
