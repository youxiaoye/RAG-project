import os
from langchain_core.messages import HumanMessage,AIMessage
from modelchoise import modelchoise
# 系统变量的设置
modelchoise.os_setenv()
#网页资源加载
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path=
    "https://baike.sogou.com/v184428933.htm"
)
documents = loader.load()
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
base_dir = r"C:\Users\sunjiaze\Desktop"    # 所有文档的存放目录
                                #声明文档列表（目前已删除）
# # 开始遍历指定文件夹
# for filename in os.listdir(base_dir):
#     # 构建完成的文件名（含有路径信息）
#     file_path = os.path.join(base_dir, filename)
#     # 分别使用不同的加载器加载各类不同的文档
#     if filename.endswith(".pdf"):
#         loader = PyPDFLoader(file_path)
#         # documents.append(loader)
#         documents.extend(loader.load())
#     elif filename.endswith(".docx"):
#         loader = Docx2txtLoader(file_path)
#         documents.extend(loader.load())


# 2. 文档（文本）切分/分割
# 2-0. 导入 字符文本切分器
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 2-1. 生成|实例化 字符文本器的实例对象
# 指定：切分文档块的大小、重叠词/Token数
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
# 完成文档切分
chunked_documents = text_splitter.split_documents(documents=documents)
print(chunked_documents)
# 3. 向量数据库存储
# Storage：将切分的文档进行向量化后 嵌入(embed)向量数据库中
# 选：Embedding： m3e
# 选：VecDB： FAISS、Qdrant
# 3-0. 生成 embedding 模型
# 导包
from langchain_huggingface import HuggingFaceEmbeddings
# 指定运算|计算设备
EMBEDDING_DEVICE = "cpu"
# 生成|实例化 embedding model
embeddings = HuggingFaceEmbeddings(model_name=r"D:\BaiduNetdiskDownload\m3e-base-huggingface",
                                   model_kwargs={'device': EMBEDDING_DEVICE})
# 3-1. embedding 到 vectordb 中
'''
    pip install qdrant-client
'''
# 导包
# from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import FAISS

# 建立索引：将词向量存储向量数据库
vectorstore = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)
# 另一个向量数据库

# 4. 创建 Retrieval 模型/链
# 4-1. 生成 llm chat_model
# 导包，baidu qianfan
from langchain_community.chat_models import ChatSparkLLM
spark_chat_model = ChatSparkLLM()
chat_mode = spark_chat_model
# completion = chat_mode.invoke(input="介绍下自己")
# print(completion.response_metadata['usage'])

# # 4-2. 生成 Retriever
# from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
# from langchain.chains import RetrievalQA # RetrievalQA链
# # 实例化一个MultiQueryRetriever
# retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=chat_mode)
#
# # 实例化一个RetrievalQA链
# qa_chain = RetrievalQA.from_chain_type(chat_mode, retriever=retriever_from_llm)

retriever = vectorstore.as_retriever()
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# 生成 ChatModel 会话的提示词
prompt = ChatPromptTemplate.from_messages([
    ("system","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])
# 生成含有历史信息的检索链
retriever_chain = create_history_aware_retriever(chat_mode, retriever, prompt)

# 继续对话，记住检索到的文档等信息
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(chat_mode, prompt)
from langchain.chains.retrieval import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

from langchain_core.messages import HumanMessage, AIMessage

# 模拟一个历史会话记录
chat_history = [
    HumanMessage(content="世界上最不好玩的游戏是什么？"),
    AIMessage(content="世界上最不好玩的游戏是永劫无间。")
]
chat_history.append(HumanMessage(content="那最好的游戏呢？"))
chat_history.append(AIMessage(content="最好的游戏也是永劫无间。"))





#数学问题agent
from langchain_community.agent_toolkits.load_tools import load_tools

tools = load_tools(tool_names=["serpapi", "llm-math"], llm=chat_mode)
from langchain.agents import (
    initialize_agent,
    AgentType
)

agent = initialize_agent(
    tools=tools,
    llm=chat_mode,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


#
#
# # 导包
# from langchain.retrievers.multi_query import MultiQueryRetriever
# # 生成|构建 MultiQueryRetriever 实例对象
# retriever = vectorstore.as_retriever()
# # chat_model 绑定
# retriever_from_llm = MultiQueryRetriever.from_llm(llm=chat_mode, retriever=retriever)
# # 导包
# from langchain.chains.retrieval_qa.base import RetrievalQA
# # 生成 QA chain
# qa_chain=RetrievalQA(retriever_from_llm=chat_mode, retriever=retriever_from_llm)
# 5. 生成 WebUI QA
'''
5.1 搭建 flask 框架项目
  模块的安装
     pip install flask
  项目框架的搭建
    static : web project 所有静态资源(pic、js、css...)
    template : web project 模板（含有 scriptlet 脚本的 html）
  完成 后端 Python 代码 和前端 html 模板的 织入（route+data）
     前端 => 后端
        跳转的目的
        携带的数据
     后端 => 前端
        什么情况下跳转到那个 html 模板
        同时带着什么数据
'''
#　导包
from flask import Flask, render_template, request,session
from flask_session import Session
app = Flask(__name__)   # 使用 Flask App
app.config['SECRET_KEY']=os.urandom(24)
app.config['SESSION_TYPE']='filesystem'
Session(app)
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []
    if request.method == 'POST':
        # 获取用户提交的名为 question 的表单域中的文本数据
        question = request.form.get('question')
        chat_history.append(HumanMessage(content=question))
        if any(char.isdigit() for char in question):
            response = agent.invoke({
                "input": question
            })
            result = response["output"]
        else:
        # retrieval_chain 回答问题
            response = response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": question
            })
            result=response["answer"]
            chat_history.append(AIMessage(content=result))
        #带着问题跳转会指定的 html 模板，同时完成渲染
        print(chat_history)
        session['history'].append({
            "role":"user",
            "content":question
        })
        session['history'].append({
            "role":"assistant",
            "content":result
        })
        #return render_template("newindex.html", result=result)
        return render_template("newindex.html", history=session['history'])
    # 如果是 GET 方式访问，直接返回 index.html（渲染返回）
    #return render_template("newindex.html")
    return render_template("newindex.html", history=session.get('history', []))
if __name__ == '__main__':
    app.run(debug=True, port=5000)