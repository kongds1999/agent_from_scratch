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

# needs_approval=True 使 Agent 在调用此工具时暂停等待审批
@function_tool(needs_approval=True)
def wait_for_human_instruction(stage_summary: str) -> str:
    """阶段任务完成后调用此工具，暂停等待人类的下一步指令。"""
    return f"人类已确认（阶段：{stage_summary}）"

agent = Agent(
    name="Researcher",
    instructions="你是一个研究助手。首先列出量子计算的三个核心概念并简要概述。列出完毕后，你必须调用 wait_for_human_instruction 工具等待人类指令。",
    model=custom_model,
    tools=[wait_for_human_instruction]
)

async def main():
    # 第一阶段：运行 Agent，遇到审批工具时自动暂停
    print("--- 开始运行 ---")
    first_run = await Runner.run(agent, "请列出量子计算的三个核心概念，然后停下来等我指令。")

    # 暂停前的模型输出保存在 new_items 中
    first_step_text = next(
        (ItemHelpers.extract_text(item.raw_item) for item in first_run.new_items if isinstance(item, MessageOutputItem)),
        None
    )
    print(f"\nAgent 回答:\n{first_step_text}")
    print(f"是否触发暂停: {bool(first_run.interruptions)}")

    # 保存检查点
    checkpoint = first_run.to_state()

    # 可选：序列化为 JSON 存入数据库
    # json_str = checkpoint.to_json()

    # 人工审核
    print("\n--- 人工审核 ---")
    await asyncio.sleep(1)

    # 批准暂停的工具调用
    if first_run.interruptions:
        checkpoint.approve(first_run.interruptions[0])

    # 追加人工指令
    human_input = input("请输入下一步指令 > ")
    checkpoint.add_input(human_input)

    # 第二阶段：从检查点恢复执行
    print("\n--- 恢复执行 ---")
    resumed_run = await Runner.run(agent, checkpoint)
    print(f"\n恢复后回答:\n{resumed_run.final_output}")

    # Token 用量按阶段独立统计
    print(f"\n第一阶段 Token: {first_run.context_wrapper.usage.total_tokens}")
    print(f"第二阶段 Token: {resumed_run.context_wrapper.usage.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())