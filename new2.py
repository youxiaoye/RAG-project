# 引入所需的模块和库
import os
import json
import langchain.chains.retrieval
import requests
import jieba
import tkinter as tk
from tkinter import filedialog, simpledialog
from bs4 import BeautifulSoup
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sympy import symbols, solve, Eq
from langchain_community.vectorstores import FAISS, Qdrant
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatSparkLLM
from langchain.prompts.chat import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from transformers import BertTokenizer, BertModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
# 定义加载本地文档的函数
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188"
os.environ["IFLYTEK_SPARK_APP_ID"] = "7972cf47"
os.environ["IFLYTEK_SPARK_API_KEY"] = "08251192f04a59184b84c6bde5103967"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "YTk2ZTBlM2RhNTdlNTAyNzRkNzEwMWQw"
os.environ["IFLYTEK_SPARK_API_URL"] = "wss://spark-api.xf-yun.com/v1.1/chat"
os.environ["IFLYTEK_SPARK_llm_DOMAIN"] = "general"
def load_local_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

# 定义加载网页文档的函数
def load_web_documents(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    return documents

vector_store =None
# 定义连接Qdrant的函数
def connect_qdrant(documents, embedding, url, api_key):
    try:
        vector_store = Qdrant.from_documents(documents=documents, embedding=embedding, url=url, api_key=api_key)
        print("成功连接Qdrant服务器.")
        return vector_store
    except Exception as e:
        print(f"连接Qdrant服务器失败: {e}")
        print("切换到本地FAISS向量数据库.")
        vector_store = FAISS.from_documents(documents=documents, embedding=embedding)
        return vector_store

# 初始化嵌入模型
embedding_device = "cpu"
embedding_model_path = r"D:\BaiduNetdiskDownload\m3e-base-huggingface"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_path, model_kwargs={'device': embedding_device})

# 创建Tkinter窗口
root = tk.Tk()
root.withdraw()

# 选择数据源
data_source = simpledialog.askstring("输入", "选择数据源 (1: 网页, 2: 本地): ")

documents = []
if data_source == "1":
    # 输入网页URL
    url_list = simpledialog.askstring("输入", "输入网页URL（多个URL用逗号分隔）:")
    if url_list:
        urls = url_list.split(',')
        documents = load_web_documents(urls)
elif data_source == "2":
    # 选择本地文件
    file_paths = filedialog.askopenfilenames(title="选择文件", filetypes=[("PDF files", "*.pdf"), ("Word files", "*.docx"), ("Text files", "*.txt")])
    if file_paths:
        documents = load_local_documents(file_paths)
else:
    print("无效的选择")
from qdrant_client import QdrantClient
# 将分词后的文本转换为Document对象
documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]

# 将文档向量化并存储在FAISS中
input2 = simpledialog.askstring("输入", "选择向量数据库 (1: FAISS, 2: Qdrant): ")

if input2 == "1":
    # 将文档向量化并存储在 FAISS 中
    vector_store = FAISS.from_documents(documents=documents, embedding=embedding)
    print("向量化存储完成（FAISS）.")
elif input2 == "2":
    vector_store = Qdrant.from_documents(documents=documents, embedding=embedding,location=":memory:",collection_name="my_documents")
else:
    raise ValueError("failed")

print("向量化存储完成.")

# 设置SparkLLM模型和对话模板
chat_model = ChatSparkLLM()
chat_model_prompt_msg_template = ChatPromptTemplate.from_template(
    """根据上下文回答问题：
    <上下文>
    {context}
    </上下文>
    问题:{input}
    代理的工作区:{agent_scratchpad}
    """
)

# 创建文档组合链
docu_chain = create_stuff_documents_chain(llm=chat_model, prompt=chat_model_prompt_msg_template)

# 创建检索链
retriever = vector_store.as_retriever()  # 使用向量存储创建检索器
retrieval_chain = langchain.chains.retrieval.create_retrieval_chain(retriever, docu_chain)

# 处理数学问题的Agent
def process_math_question(question):
    try:
        # 使用 sympy 解决简单的数学方程，例如: "2x + 3 = 7"
        equation = question
        # 定义变量
        var = symbols('x')
        # 创建等式
        eq = Eq(eval(equation.split('=')[0].strip()), eval(equation.split('=')[1].strip()))
        # 解方程
        solution = solve(eq, var)
        return f"解答是: {solution[0]}"
    except Exception as e:
        # 如果出问题，返回错误信息
        return f"无法解答此数学问题: {e}"

math_tool = Tool(
    name="process_math",
    func=process_math_question,
    description="解决简单问题"
)

tool1 = [math_tool]
agent1 = create_openai_functions_agent(
    tools=[math_tool],
    llm=chat_model,
    prompt=chat_model_prompt_msg_template,
)

agent_executor1 = AgentExecutor(agent=agent1, tools=tool1, verbose=True)



# 开始对话循环
chat_history = []  # 初始化对话历史记录列表

print("开始对话（输入 'end' 结束）：")
while True:
    human_message = input("请输入问题：").strip().lower()
    if human_message == "end":
        break

    invoke_input = {
        "input": human_message,
        "context": {},
        "chat_history": [(msg.content, "human" if isinstance(msg, HumanMessage) else "ai") for msg in chat_history],
        "agent_scratchpad": {}
    }

    # 获取用户输入的问题
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": human_message,
        "agent_scratchpad": {}
    })

    if '=' in human_message:
        ai_message = agent_executor1.invoke(input=invoke_input)
    else:

        ai_message =response.get("answer", "not found")

    print(ai_message)
    # 手动追加聊天记录
    chat_history.append(HumanMessage(human_message))
    chat_history.append(AIMessage(str(ai_message)))

# 生成时间戳并写入文件
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"chat_history_{timestamp}.txt"
desktop_path = r"C:\Users\sunjiaze\Desktop"
file_path = os.path.join(desktop_path, filename)

with open(file_path, "w", encoding="utf-8") as file:
    for message in chat_history:
        role = "用户" if isinstance(message, HumanMessage) else "AI"
        file.write(f"{role}: {message.content}\n")

print(f"聊天记录已保存至 {file_path}")
print("对话结束.")
