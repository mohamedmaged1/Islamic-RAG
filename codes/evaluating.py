import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings  # Changed this
from langchain import hub
import bs4
from langchain_core.prompts import ChatPromptTemplate


# Load your CSV
df = pd.read_csv(r".\Data\Book.csv") # Must contain: question, reference_answer, generated_answer

# Your prompt
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """أنت عالم فقيه متخصص في الشريعة الإسلامية، ومهمتك تقييم الإجابات على المسائل الفقهية.

ستُعطى العناصر التالية:
-  سؤال فقهي
- إجابة مأخوذة من كتاب موثوق
- إجابة مولدة من نموذج ذكاء اصطناعي

قارن بين الإجابتين من حيث:
1. التوافق مع الحكم المذكور في الكتاب
2. صحة ومصداقية المحتوى
3. وضوح العرض والأسلوب

ثم قدم تقييمك وفق المعايير التالية:

-  هل تتفق الإجابة المولدة مع الحكم الوارد في الكتاب؟ (نعم / لا)
-  إذا كانت لا تتفق، ما هو الخطأ؟ وما التصحيح؟
-  ما درجة الثقة في الإجابة المولدة؟ (من 1 إلى 5)
-  هل يمكن اعتمادها كجواب فقهي موثوق؟ (نعم مع تعديلات / جزئياً / لا)"""),

    ("human", "البيانات:\nالسؤال: {question}\n\nإجابة الكتاب:\n{reference_answer}\n\nإجابة النموذج:\n{generated_answer}")
])

# Create the prompt template
prompt = eval_prompt

# LLM setup (adjust for your OpenAI, DashScope, or other provider)
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=dashscope_api_key,
    openai_api_base=dashscope_base_url,
    temperature=0)

# Output parser
parser = StrOutputParser()

# Evaluate row-by-row
results = []
for idx, row in df.iterrows():
    question = row["Question"]
    reference_answer = row["GT"]
    generated_answer = row["Response"]
    
    try:
        # Create the full input to LLM
        messages = prompt.format_messages(
            question=question, 
            reference_answer=reference_answer, 
            generated_answer=generated_answer
        )
        
        response = llm.invoke(messages)
        evaluated = parser.parse(response)
        
        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "evaluation": evaluated
        })

        print(f"✅ Evaluated {idx + 1}/{len(df)}")
    
    except Exception as e:
        print(f"❌ Error at row {idx}: {e}")
        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "evaluation": f"ERROR: {str(e)}"
        })

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("evaluated_fiqh_responses.csv", index=False, encoding="utf-8-sig")

print("✅ Evaluation completed and saved.")