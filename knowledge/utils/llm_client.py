import os,logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()

cache_llm_client={}

def get_llm_client(mode_name:str=None, temperature:float=0.0, response_format:bool=False):
    """

    Returns:  返回LLM客户端对象
    #
    缓存的对象是： client
    缓存的Key：不同的节点用不同的模型以及同一个节点用不同的响应格式
    """

    # 1. 获取模型的名字
    model_name = mode_name or os.getenv("ITEM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    cache_key = (mode_name, response_format) # 复合缓存Key

    #2.缓存命中 直接返回
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]

    #3. 返回的内容格式
    model_kwargs = {}
    if response_format:
        model_kwargs["response_format"] = {'type': 'json_object'}
    try:
        # 4. 定义模型实例
        client = ChatOpenAI(
            model = model_name,
            api_key = api_key,
            base_url = api_base,
            temperature=0,
            extra_body={"enable_thinking":False},
            model_kwargs = model_kwargs
        )

        #5. 同步数据
        cache_llm_client[cache_key] = client

        return client
    except Exception as e:
        logger.error(f"LLM client error: {str(e)}")
        return None


if __name__ == '__main__':

    llm_client = get_llm_client(mode_name="kimi-k2.5")

    import json

    # ai_message = llm_client.invoke("你好，请问您是谁?")
    # 使用本质发送请求（底层将model_kwargs的所有参数都在发送请求之前拼接到请求体身上）
    ai_message = llm_client.invoke("您好，请给我讲一个笑话，返回json格式：{\"key\":\"value\"}")
    print(ai_message.content)

    # json对象 json字符串
    # json_object = json.loads(ai_message.content)