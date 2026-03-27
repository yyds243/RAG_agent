from anthropic import Anthropic
from langchain_openai import ChatOpenAI

def get_opus_client():
    client = Anthropic(
        base_url="https://timesniper.club",
        api_key="sk-0l8NOzKsAGPlRVPVikA0nJ5hBtmDQIuWNVhiQS08VPDYEBJT"
    )

    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role":"user",
                "content":"你是谁，能做什么",
            }
        ],
        model="claude-opus-4-5-20251101-thinking",
    )
    return message


def get_llmclient():

    client = ChatOpenAI(
        base_url= "https://timesniper.club",
        api_key="sk-tWYv8p6fD8GjT4eSAsWDn1lUcfavJMNAoePBf6LJwcqriZA6",
        model="claude-opus-4-5-20251101-thinking",
    )
    return client

if __name__ == '__main__':
    llm_client = get_opus_client()
    print(llm_client.content)