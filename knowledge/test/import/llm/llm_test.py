import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# 打印当前工作目录，确认 .env 应该放在哪里
print(f"当前工作目录: {os.getcwd()}")

# 加载环境变量
load_dotenv()

# 【关键调试步骤】打印读取到的值
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("ITEM_MODEL")

print(f"读取到的 API Key: {api_key}") # 如果这里打印 None 或空，说明没读到
print(f"读取到的 API Base: {api_base}")
print(f"读取到的 Model: {model_name}")

if not api_key:
    raise Exception("错误：未能从环境变量中读取到 OPENAI_API_KEY！请检查 .env 文件是否存在及内容格式。")

def get_llm_client():
    """

    Returns:  返回LLM客户端对象
    #
    缓存的对象是： client
    缓存的Key：不同的节点用不同的模型以及同一个节点用不同的响应格式
    """

    # 1. 获取模型的名字
    model_name = os.getenv("ITEM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    # 2. 定义模型实例
    client = ChatOpenAI(
        model = model_name,
        api_key = api_key,
        base_url = api_base,
        temperature=0,
        extra_body={"enable_thinking":False},
        timeout=30
    )

    return client

if __name__ == '__main__':
    llm_client = get_llm_client(mode)

    ai_message = llm_client.invoke("你好请问你是谁")

    print(ai_message.content)