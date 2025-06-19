import os
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings  # Changed this
from langchain_core.prompts import ChatPromptTemplate
import pickle
from helpers.config import get_settings,Settings



os.environ["LANGSMITH_TRACING"] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "your_api_key_here"
# Configure API key

dashscope_api_key = get_settings().DASHSCOPE_API_KEY
dashscope_base_url = get_settings().DASHSCOPE_BASE_URL


with open("./Data/all_splits.pkl", "rb") as f:
    all_splits = pickle.load(f)



# Build your own system prompt in Arabic or adjusted tone
prompt = ChatPromptTemplate.from_messages([
    ("system",  " أنت عالم فقه مسلم فى غايه الذكاء مقدم لك بيانات لتستمد منها اجاباتك بناء على سؤال مقدم لك واعلم انه يوجد ايات قرءانية مكتوبه بطريقه خطأ لا تستمد منها أيضا تجنب الحروف غير العربية"),
    ("human", "استخدم السياق التالي للإجابة على السؤال:\n\n{context}\n\nالسؤال: {question}\n\nالإجابة:")
])



embeddings = DashScopeEmbeddings(
    dashscope_api_key=dashscope_api_key,
    model="text-embedding-v1"  
)

# Create vectorstore
vector_store = Chroma(embedding_function=embeddings)
_ = vector_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever(   
    search_type="similarity",    # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 1}       # how many results to retrieve
)


# Configure LLM for DashScope
llm = ChatOpenAI(
    model_name="qwen-plus",  # or qwen-Turpo
    openai_api_key=dashscope_api_key,
    openai_api_base=dashscope_base_url,
    temperature=0
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Example question
text = "رد على من يطعن فى حجيه السنة "
response = rag_chain.invoke(text)
print(response)
