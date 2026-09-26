import os, sys
from dotenv import load_dotenv
from agents import Agent, Runner, set_tracing_disabled, set_default_openai_api
from tools import run_bash, read_file, write_file, run_grep, run_glob, revert_file

load_dotenv() 
set_tracing_disabled(True)  # Disable tracing for cleaner output
set_default_openai_api("chat_completions")  # 兼容 LiteLLM / 第三方中转，避免 Responses API 加密凭证跨节点校验报错


# 3. 终端交互 REPL                                                                                                     
def main():
    
    agent = Agent(
        name="CodingAssistant",
        model="openai/gpt-5.5",
        instructions=f"You are a coding agent at {os.getcwd()}. Use the bash tool to solve tasks. Act, don't explain unless needed.",
        tools=[run_bash, read_file, write_file, run_grep, run_glob, revert_file],
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
        result = Runner.run_sync(agent, input=current_input)
                                                                                                                        
        # 更新多轮对话历史                                                                                             
        conversation_history = result.to_input_list()                                                                  
                                                                                                                        
        print("\n\033[32mFinal Answer:\033[0m")                                                                        
        print(result.final_output)                                                                                     
        print()                                                                                                        
                                                                                                                        
if __name__ == "__main__":                                                                                             
        main()