from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pickle
import os

# ========== Configuration ==========
DASHSCOPE_API_KEY = "<>"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ========= Load Data =========
with open("./Data/all_splits.pkl", "rb") as f:
    all_splits = pickle.load(f)

# ========= Prompt Template =========
prompt = ChatPromptTemplate.from_messages([
    ("system", """أنت عالم فقه مسلم ذكي. استخدم السياق التالي للإجابة على السؤال. تجنب الحروف غير العربية وتجاهل الآيات المكتوبة بطريقة غير صحيحة."""),
    ("human", "السياق:\n\n{context}\n\nالسؤال: {question}\n\nالإجابة:")
])

# ========= Embedding & Vectorstore Setup =========
embeddings = DashScopeEmbeddings(
    dashscope_api_key=DASHSCOPE_API_KEY,
    model="text-embedding-v1"
)

# Create vectorstore
vector_store = Chroma(embedding_function=embeddings)
_ = vector_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 1}
)

# ========= LLM Setup =========
llm = ChatOpenAI(
    model_name="deepseek-r1",  # deepseek-r1 or adjust if different
    openai_api_key=DASHSCOPE_API_KEY,
    openai_api_base=DASHSCOPE_BASE_URL,
    temperature=0,
    extra_body={"enable_thinking": False}
)

# ========= Helper Functions =========
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ========= RAG Chain =========
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========= FastAPI Setup =========
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(question: str):
    try:
        # Try main RAG pipeline
        answer = rag_chain.invoke(question)
        return answer#{"answer": answer}
    except Exception:
        try:
            # Fallback: Without context
            prompt_fallback = ChatPromptTemplate.from_messages([
                ("system", "أنت عالم فقه مسلم ذكي. قدّم إجابة معتمداً على معرفتك إذا لم يُقدّم لك سياق."),
                ("human", "{question}")
            ])
            fallback_chain = prompt_fallback | llm | StrOutputParser()
            fallback_answer = fallback_chain.invoke({"question": question})
            return  fallback_answer
        except Exception as e:
            return {"answer": "عذرًا، لا يمكنني الإجابة على هذا السؤال لأنه حساس أو خارج النطاق المسموح به."}
        


