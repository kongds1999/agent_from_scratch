#!/usr/bin/env python3
"""
s03_todo_write.py: 实现代理式规划和状态跟踪。
Motto: "An agent without a plan drifts"

这个模块引入了一个任务管理系统，Agent 必须使用它来组织复杂的多步操作。
通过将计划持久化到 JSON 文件中，代理创建了一个“真相来源”，
这显著减少了长时间任务中的幻觉和逻辑错误。

New Capabilities:
    1. todo_write: 将任务描述列表序列化为 JSON 文件。
    2. todo_read: 从 JSON 文件反序列化并显示当前计划状态。
    3. todo_update: 修改特定任务的状态（待办 -> 已完成）。

这个会话的核心是“系统提示”，它强制 Agent 使用这些工具，
有效地创建了一个“计划-执行”循环。
"""

# === Standard Library Imports ===
import os      # Operating system interfaces (for pathing and environment)
import json    # JSON encoding and decoding for the todo file
import sys     # System-specific parameters (for exiting)
from typing import List, Dict, Any, Union, Optional  # Type hinting for robust code
from agents import Agent, Runner, set_tracing_disabled  # Core agent framework
from agents.exceptions import MaxTurnsExceeded  # 超过 max_turns 时抛出
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from tools import AGENT_TOOLS
from dotenv import load_dotenv  # Load environment variables from .env file
load_dotenv() 
set_tracing_disabled(True)

# Specialized System Prompt: This is the "Policy" the agent follows.
# It explicitly instructs the model to create a plan before acting.
SYSTEM: str = (
    f"You are a coding agent at {os.getcwd()}. "
    "Before working on any multi-step task, ALWAYS call todo_write first "
    "to write your plan. Then execute each step and call todo_update after each one. "
    "This ensures you stay on track and don't skip steps."
)

# === Main Execution Block ===

# 3. 终端交互 REPL                                                                                                     
async def main():
    print("\033[90ms03: plan before execute | todo_write + todo_update\033[0m\n")
    agent = Agent(
        name="CodingAssistant",
        model="deepseek-v4-flash-vision-exp",
        instructions=f"You are a coding agent at {os.getcwd()}. Use the bash tool to solve tasks. Act, don't explain unless needed.",
        tools=AGENT_TOOLS,
    )
    print("\033[90m openai-agents-code v0.1.0 \033[0m")                                                                
                                                                                                                        
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
                                                                                                                        
        # Runner 内部会自动处理多轮 Tool Call 循环
        # max_turns 默认是 10（agents.run_config.DEFAULT_MAX_TURNS），多步任务很容易超；
        # 传 None 可完全关闭限制。
        result = Runner.run_streamed(
            agent, 
            input=current_input, 
            max_turns=50
        )
        print("\n\033[32mFinal Answer:\033[0m")
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    result_text = event.data.delta
                    if result_text:
                        print(event.data.delta, end="", flush=True)
        except MaxTurnsExceeded as e:
            print(f"\n\033[31m[!] {e} — 增大 max_turns 或传 None\033[0m")
        # result = Runner.run_sync(agent, input=current_input)
                                                                                                                        
        # 更新多轮对话历史                                                                                             
        conversation_history = result.to_input_list()                                                                  
        print()  


if __name__ == "__main__":
    # Standard entry point execution
    asyncio.run(main())