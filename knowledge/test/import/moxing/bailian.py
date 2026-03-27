from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(
    api_key = "sk-fa99f371d1cd4105b0ead0f75e0f7cd6",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"
                    },
                },
                {"type": "text", "text": "这道题怎么解答？"},
            ],
        },
    ],
    # stream=True,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    # extra_body={
    #     'enable_thinking': True,
    #     "thinking_budget": 81920},

    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
)
answer = completion.choices[0].message.content
print(answer)