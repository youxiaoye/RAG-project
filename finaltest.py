import mimetypes
import os
import json
import hashlib
import pickle
import sys
import time
import easyocr
import redis
import pymysql
from typing import Dict, List, Any
import asyncio
import uuid
import edge_tts
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from fastapi import FastAPI
from modelchoise import modelchoise
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, text
from contextlib import contextmanager
from flask_session import Session as FlaskSession
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from flask_socketio import SocketIO, emit
from fastapi.responses import StreamingResponse
import docx
import PyPDF2
import requests
from bs4 import BeautifulSoup
from typing import List
from flask import Response,stream_with_context
import subprocess
import speech_recognition as sr
from io import BytesIO

#使用语音输入服务需要科学上网
# 系统变量的设置
app = Flask(__name__)
modelchoise.os_setenv()
vectorstore =None

loader = WebBaseLoader(web_path="https://baike.sogou.com/v184428933.htm")
documents = loader.load()

base_dir = r"C:\\Users\\sunjiaze\\Desktop"

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents=documents)
print(chunked_documents)

# 向量数据库存储
EMBEDDING_DEVICE = "cpu"
embeddings = HuggingFaceEmbeddings(model_name=r"D:\\BaiduNetdiskDownload\\m3e-base-huggingface",
                                   model_kwargs={'device': EMBEDDING_DEVICE})
vectorstore = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)

# 生成 ChatModel 会话的提示词
chat_mode = modelchoise.getsparkllm()


retriever = vectorstore.as_retriever()
# def load_vectorstore(file_path):
#     if os.path.exists(file_path):
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#     return None
#
# vectorstore_file_path = 'vectorstore.pkl'
# vectorstore = load_vectorstore(vectorstore_file_path)
#

#提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])
#生成检索链
retriever_chain = create_history_aware_retriever(chat_mode, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(chat_mode, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
#模拟历史对话（历史对话实例化）
chat_history = [
    HumanMessage(content="世界上最不好玩的游戏是什么？"),
    AIMessage(content="世界上最不好玩的游戏是永劫无间。")
]
chat_history.append(HumanMessage(content="那最好的游戏呢？"))
chat_history.append(AIMessage(content="最好的游戏也是永劫无间。"))

# 数学问题agent
# tools = load_tools(tool_names=["serpapi", "llm-math"], llm=chat_mode)
# agent = initialize_agent(
#     tools=tools,
#     llm=chat_mode,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True
# )

################################################################

ocr_reader = easyocr.Reader(['ch_sim', 'en'])
################################################################
# MySQL 配置
HOSTNAME = "127.0.0.1"
PORT = 3306
USERNAME = "root"
PASSWORD = "sjz040705"
DATABASE = "dbsclab2018"

db_url = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
    USERNAME,
    PASSWORD,
    HOSTNAME,
    PORT,
    DATABASE
)

engine = create_engine(db_url)
Base = declarative_base()
SessionLocal = sessionmaker(engine)
scoped_session = SessionLocal()

# 定义 SQLAlchemy 模型
class User(Base):
    __tablename__ = 'langchain_final_project_users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(200), nullable=False)
    nickname = Column(String(50), nullable=False)
# 创建表结构
Base.metadata.create_all(engine)

# Redis 配置
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'session:'
app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379, db=0)
flask_session = FlaskSession(app)

#ocr配置
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
# 检查数据库连接状况
def check_db_connection():
    try:
        conn = engine.connect()
        res = conn.execute(text("SELECT 1"))
        conn.close()
        return True
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False

# 问题回答功能（使用 LLM 模型）
@contextmanager
def session_scope():
    """Provide a session scope."""
    connection = engine.connect()
    transaction = connection.begin()
    options = dict(bind=connection, binds={})
    db_session = SessionLocal(**options)
    try:
        yield db_session
        transaction.commit()
    except:
        transaction.rollback()
        raise
    finally:
        db_session.close()
        connection.close()

# 登陆界面处理
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not check_db_connection():
            flash('数据库连接失败，请稍后再试。')
            return render_template('login.html')

        username = request.form['username']
        password = request.form['password']

        with engine.connect() as conn:
            query = text("SELECT * FROM langchain_final_project_users WHERE username = :username AND password = :password")
            user = conn.execute(query, {"username": username, "password": password}).fetchone()
            conn.close()
            if user:
                session['logged_in'] = True
                session['user_id'] = user.id
                session['nickname'] = user.nickname
                session.modified = True
                return redirect(url_for('index'))
            else:
                flash('Invalid credentials. Please try again.')

    return render_template('login.html')

# 注册界面处理
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if not check_db_connection():
            flash('数据库连接失败，请稍后再试。')
            return render_template('register.html')

        username = request.form['username']
        password = request.form['password']
        nickname = request.form['nickname']

        try:
            with session_scope() as session:
                # 检查用户名是否已存在
                existing_user = session.query(User).filter_by(username=username).first()
                if existing_user:
                    flash('用户名已存在，请选择其他用户名。')
                    return render_template('register.html')

                # 创建新用户
                new_user = User(username=username, password=password, nickname=nickname)
                session.add(new_user)
                session.commit()

            flash('注册成功，请登录。')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"注册失败: {e}")
            flash('注册失败，请重试。')
            return render_template('register.html')

    return render_template('register.html')

# 登出处理
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('index', some_param='value'))

@app.route('/failed', methods=['GET', 'POST'])
def failed():
    return render_template('failed.html')

@app.route('/test_redirect',methods=['GET', 'POST'])
def test_redirect():
    return render_template('newindex.html')

@app.route('/path-to-your-endpoint', methods=['GET'])
def ajax_redirect():
    # 返回 JSON 响应，包含重定向 URL
    return jsonify({'redirect': url_for('index')})

# 主页处理（包含问题回答功能）
from flask import Response
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'logged_in' not in session:
        session['question_count'] = session.get('question_count', 0)
        if session['question_count'] >= 3:
            flash('您已达到未登录用户的提问限制，请登录以继续提问。')
            return redirect(url_for('failed'))

    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # 使用 easyocr 进行 OCR
                ocr_result = ocr_reader.readtext(filepath)
                extracted_text = " ".join([text[1] for text in ocr_result])

                question = f"总结这段文本内容并且发送给我: {extracted_text}"
                session['history'].append({"role": "user", "content": f"图片: {filename}"})
            else:
                question = request.form.get('question')
                session['history'].append({
                    "role": "user",
                    "content": question
                })
        else:
            question = request.form.get('question')
            session['history'].append({
                "role": "user",
                "content": question
            })

        if question:
            session['question_count'] = session.get('question_count', 0) + 1
            chat_history.append(HumanMessage(content=question))
            response = retrieval_chain.invoke({"chat_history": chat_history, "input": question})
            result = response["answer"]
            chat_history.append(AIMessage(content=result))
            session['history'].append({"role": "assistant", "content": result})

            session.modified = True
    return render_template("newindex.html", history=session['history'], user_info=session.get('user_id'))


# def generate_response_stream(question):
#     output = {}
#     curr_key = None
#     for chunk in retrieval_chain.stream({"chat_history": chat_history, "input": question}):
#         for key in chunk:
#             if key not in output:
#                 output[key] = chunk[key]
#             else:
#                 output[key] += chunk[key]
#             if key != curr_key:
#                 yield json.dumps(f"\n\n{key}: {chunk[key]}")
#                 print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
#             else:
#                 yield json.dumps(chunk[key])
#                 print(chunk[key], end="", flush=True)
#             curr_key = key
#         time.sleep(0.05)  # 模拟生成内容的延迟
#
# @app.route('/stream', methods=['GET','POST'])
# def stream_response():
#     if 'history' not in session:
#         session['history'] = []
#
#     if request.method == 'POST':
#         if 'image' in request.files:
#             file = request.files['image']
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(filepath)
#
#                 # 使用 easyocr 进行 OCR
#                 ocr_result = ocr_reader.readtext(filepath)
#                 extracted_text = " ".join([text[1] for text in ocr_result])
#
#                 question = f"总结这段文本内容并且发送给我: {extracted_text}"
#                 session['history'].append({"role": "user", "content": f"图片: {filename}"})
#             else:
#                 question = request.form.get('question')
#                 session['history'].append({
#                     "role": "user",
#                     "content": question
#                 })
#         else:
#             question = request.form.get('question')
#             session['history'].append({
#                 "role": "user",
#                 "content": question
#             })
#
#         # 增加问题到 chat_history
#         question = request.form.get('question')
#         session['question_count'] = session.get('question_count', 0) + 1
#         chat_history.append(HumanMessage(content=question))
#
#         for chunk in retrieval_chain.stream({"chat_history": chat_history, "input": question}):
#             chat_history.append(AIMessage(content=str(chunk)))
#
#         session.modified = True
#
#         # 使用流式输出生成器函数
#         return Response(generate_response_stream(question), content_type='text/event-stream')
#     return "Streaming endpoint ready to accept POST requests."
#
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if 'logged_in' not in session:
#         session['question_count'] = session.get('question_count', 0)
#         if session['question_count'] >= 3:
#             flash('您已达到未登录用户的提问限制，请登录以继续提问。')
#             return redirect(url_for('failed'))
#
#     if 'history' not in session:
#         session['history'] = []
#
#     if request.method == 'POST':
#         return redirect(url_for('stream_response'))
#     # 如果是 GET 请求，渲染页面
#     return render_template("newindex.html", history=session['history'], user_info=session.get('user_id'))




@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}),400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 使用 easyocr 进行 OCR
        ocr_result = ocr_reader.readtext(filepath)
        extracted_text = " ".join([text[1] for text in ocr_result])

        return jsonify({'results': extracted_text})
    else:
        return jsonify({'error': 'Invalid file format'}), 400

async def generate_tts(text: str, file_path: str):
    try:
        communicate = edge_tts.Communicate(text, voice="zh-CN-YunyangNeural")
        await communicate.save(file_path)
        print(f"Generated TTS file at: {file_path}")
    except Exception as e:
        print(f"Error generating TTS file: {e}")
def save_tts(text: str, file_path: str):
    asyncio.run(generate_tts(text, file_path))
@app.route('/tts', methods=['POST'])
def tts():
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # 确保 static 目录存在
    if not os.path.exists('static'):
        os.makedirs('static')

    timestamp =int(round(time.time()*1000))

    # 生成语音文件
    unique_filename = f'{timestamp}.mp3'
    file_path = os.path.join('static', unique_filename)

    try:
        save_tts(text, file_path)
    except Exception as e:
        return jsonify({'error': f"Failed to generate TTS: {e}"}), 500

    # 生成音频文件的 URL
    audio_url = '/static/' + unique_filename

    return jsonify({'audio_url': audio_url})

def extract_text_from_pdf(file_stream):
    text = ""
    reader = PyPDF2.PdfReader(file_stream)
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text += page.extract_text()
    return text
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def guess_mime_type(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        # 手动指定一些常见的 MIME 类型
        if filename.lower().endswith('.pdf'):
            mime_type = 'application/pdf'
        elif filename.lower().endswith('.docx'):
            mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            mime_type = 'unknown'
    return mime_type
def check_file_type(file_stream):
    file_stream.seek(0)
    header = file_stream.read(4)

    if header.startswith(b'%PDF'):
        return 'application/pdf'
    elif header.startswith(b'PK\x03\x04'):
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    else:
        return 'unknown'
@app.route('/upload', methods=['POST'])
def upload():
    if 'history' not in session:
        session['history'] = []

    if 'file' not in request.files:
        flash('未选择文件')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('未选择文件')
        return redirect(request.url)

    if file:
        file_stream = BytesIO(file.read())
        filename = secure_filename(file.filename)

        mime_type = guess_mime_type(filename)
        if mime_type == 'unknown':
            mime_type = check_file_type(file_stream)

        print(f"文件 MIME 类型: {mime_type}")

        if mime_type == 'application/pdf':
            file_stream.seek(0)  # 重置流位置
            text = extract_text_from_pdf(file_stream)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            file_stream.seek(0)  # 重置流位置
            text = extract_text_from_docx(file_stream)
        else:
            text = "Unsupported file format."

        print(f"提取的文本: {text}")

        # 使用聊天模型生成文本总结
        response = retrieval_chain.invoke({"chat_history": chat_history, "input": text})
        summary = response["answer"]

        # 更新会话历史记录
        session['history'].append({
            "role": "user",
            "content": f"上传的文件: {filename}"
        })
        session['history'].append({
            "role": "assistant",
            "content": summary
        })
        session.modified = True

        # 渲染并返回结果页面
        return render_template("newindex.html", history=session['history'])

    return redirect(url_for('index'))
def process_webpages(urls: List[str]):
    all_texts = []

    for url in urls:
        try:
            # 发起请求并解析网页
            response = requests.get(url)
            response.raise_for_status()  # 确保请求成功
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取文本内容
            page_text = soup.get_text(separator=' ', strip=True)
            all_texts.append(page_text)
            print(f"Processed URL: {url}")
            print(page_text)
        except requests.RequestException as e:
            print(f"Error processing URL {url}: {e}")

    # 将所有网页文本合并成一个字符串
    combined_text = " ".join(all_texts)
    print("所有网页处理完毕。")
    print(combined_text)
    return combined_text

@app.route('/spider', methods=['GET', 'POST'])
def webview():
    if 'urls' not in request.form:
        return jsonify({'error': 'No URLs provided'}), 400

    urls = request.form.get('urls').split(',')
    urls = [url.strip() for url in urls if url.strip()]
    if not urls:
        return jsonify({'error': 'No valid URLs provided'}), 400

    # 处理多个网页
    combined_text= process_webpages(urls)
    question = f"总结这段文本内容并且发送给我: {combined_text}"
    session['history'].append({"role": "user", "content": f"爬取网址: {urls}"})

    response = retrieval_chain.invoke({"chat_history": session['history'], "input": question})
    result = response["answer"]

    # 返回“等待下一步指示”
    session['history'].append({
        "role": "assistant",
        "content": result
    })
    return render_template("newindex.html", history=session['history'], user_info=session.get('user_id'))

@app.route('/audio', methods=['POST'])
def audio():
    if 'history' not in session:
        session['history'] = []
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    timestamp = int(time.time())
    original_filename = audio_file.filename
    base, ext = os.path.splitext(original_filename)
    unique_filename = f"{base}_{timestamp}{ext}"
    audio_path = os.path.join('audio', unique_filename)
    audio_file.save(audio_path)

    wav_path = os.path.splitext(audio_path)[0] + '.wav'
    ffmpeg_path = r"D:\ffmpeg\ffmpeg-6.1.1-full_build\bin"
    try:
        # 加载音频文件
        audio = AudioSegment.from_file(audio_path, format="webm")
        # 导出为wav格式
        wav_path = os.path.splitext(audio_path)[0] + '.wav'
        audio.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 使用ffmpeg将webm转换为wav格式
    # try:
    #     subprocess.run([ffmpeg_path, '-i', audio_path, wav_path], check=True)
    # except subprocess.CalledProcessError as e:
    #     return jsonify({"error": str(e)}), 500

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            question = recognizer.recognize_google(audio_data, language='zh-CN')
        except sr.UnknownValueError:
            question = "无法识别音频内容"
        except sr.RequestError:
            question = "语音识别服务出错"
    if question:
        response = retrieval_chain.invoke({"chat_history": chat_history, "input": question})
        result = response["answer"]
        session['history'].append({"role": "user", "content": question})
        session['history'].append({"role": "assistant", "content": result})
        return jsonify({'question': question, 'result': result})
    return  render_template("newindex.html", history=session['history'], user_info=session.get('user_id'))

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    sessions = os.listdir('history')
    return jsonify(sessions)

@app.route('/load_session/<int:session_id>', methods=['GET'])
def load_session(session_id):
    session_file = f'history/session_{session_id}.html'
    with open(session_file, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

@app.route('/save_session', methods=['POST'])
def save_session():
    session_dir = 'history'
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    session_files = os.listdir(session_dir)
    session_id = len(session_files)
    session_file = os.path.join(session_dir, f'session_{session_id}.html')
    with open(session_file, 'w', encoding='utf-8') as file:
        file.write(render_template("newindex.html", history=session['history']))
    session['history'] = []  # 清空当前会话历史
    return '', 204
if __name__ == '__main__':
    if check_db_connection():
        print("success!")
    else:
        print("failed!")


    app.run(debug=True, port=5000)
