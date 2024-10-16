import os
from langchain_community.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
def os_setenv():
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188"
    # os.environ["IFLYTEK_SPARK_APP_ID"] = "4a62342f"
    # os.environ["IFLYTEK_SPARK_API_KEY"] = "cbf123326396efef53d831f311fa2ab0"
    # os.environ["IFLYTEK_SPARK_API_SECRET"] = "YjhjMGJhZTI0OWViZGZlODE1NWI5YjM1"
    # os.environ["IFLYTEK_SPARK_API_URL"] = "wss://spark-api.xf-yun.com/v1.1/chat"
    # os.environ["IFLYTEK_SPARK_llm_DOMAIN"] = "general"
    os.environ["OPENAI_API_KEY"] = "sk-NzFPhk6KmbRwrF7uwXKclG4eiyB2dOr4Fby1HMgossbs35yl"
    os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"
    os.environ["SERPAPI_API_KEY"]="a8e2ef2c1fbadd4c2ddb6ff3595f5fd137c0c9265e75b5f53695e68f845c02a3"
    # 请求网页内容
def getsparkllm():
    chatmodel = OpenAI()
    return chatmodel

