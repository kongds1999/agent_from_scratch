import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, set_tracing_disabled, function_tool
from agents.items import ItemHelpers, MessageOutputItem
from agents.extensions.models.litellm_model import LitellmModel

load_dotenv()
set_tracing_disabled(True)

custom_model = LitellmModel(
    model="gpt-5.6-terra",
    base_url=os.getenv("LITELLM_PROXY_API_BASE", ""),
    api_key=os.getenv("LITELLM_PROXY_API_KEY", "")
)

# 1. 声明一个需要人工审批/确认的工具（触发 RunState 中断挂起，使 add_input 有效）
@function_tool(needs_approval=True)
def wait_for_human_instruction(stage_summary: str) -> str:
    """当阶段任务完成时调用此工具停下来，向人类汇报并等待进一步指令。"""
    return f"人类已确认并放行（阶段：{stage_summary}）"

# 2. 定义研究助手 Agent
agent = Agent(
    name="Researcher",
    instructions="你是一个研究助手。首先列出量子计算的三个核心概念并简要概述。列出完毕后，你必须调用 wait_for_human_instruction 工具等待人类指令。",
    model=custom_model,
    tools=[wait_for_human_instruction]
)

async def main():
    # ==== 第一阶段：初次运行并触发中断建立检查点 ====
    print("--- 启动初次运行 ---")
    query = "请帮我列出量子计算的三个核心概念，并停下来等待我的指令。"
    print(f"> {query}")
    first_run = await Runner.run(agent, query)
    # 1. 提取初次运行（中断挂起前）模型生成的回答文本
    first_step_text = next(
        (ItemHelpers.extract_text(item.raw_item) for item in first_run.new_items if isinstance(item, MessageOutputItem)),
        None
    )
    print(f"\n初次运行回答:\n{first_step_text}")
    print(f"初次运行是否触发中断: {bool(first_run.interruptions)}")
    if first_run.interruptions:
        print(f"待审批工具调用: {first_run.interruptions[0].tool_name}")
    
    # 将当前的执行状态捕获为快照（Checkpoint RunState）
    checkpoint_state = first_run.to_state()
    
    # 此时你可以将 checkpoint_state 序列化为 JSON 字符串存入数据库 (如 Redis/PostgreSQL)
    # json_data = checkpoint_state.to_json()
    # print(f"\n--- 检查点已保存为 JSON (长度: {len(json_data)}) ---")

    # 1. 批准挂起的中断
    if first_run.interruptions:
        checkpoint_state.approve(first_run.interruptions[0])

    # 2. 向处于中断状态的 RunState 阶段性追加入工输入或审批反馈
    checkpoint_state.add_input(input("> "))

    # ==== 第二阶段：从检查点恢复并继续执行 ====
    print("\n--- 从检查点恢复执行 ---")
    resumed_run = await Runner.run(agent, checkpoint_state)
    print(f"\n恢复后的追加回答:\n{resumed_run.final_output}")
    
    # 验证 Token 及 Usage 隔离
    print(f"\n初次运行总 Token: {first_run.context_wrapper.usage.total_tokens}")
    print(f"恢复后该段独立消耗 Token: {resumed_run.context_wrapper.usage.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
