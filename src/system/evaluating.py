import pandas as pd
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

# Load your CSV
dashscope_api_key = "<your_dashscope_api_key_here>"
dashscope_base_url = "<your_dashscope_base_url_here>"


df = pd.read_csv(r"./Data/combined_Alldataset_with_models.csv") # Must contain: question, reference_answer, generated_answer

# Your prompt


eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """Your task is only to evaluate the answer in range from 1 to 5. 
     You will be given a question, a reference answer from a trusted book, and a generated answer from an AI model.
     Please evaluate implicitly the generated answer based on its accuracy, relevance, and clarity compared to the reference answer. And only RETURN NUMBER FROM 1 TO 5."""),

    ("human", "Data:\nQuestion: {question}\n\nReference Answer:\n{reference_answer}\n\nGenerated Answer:\n{generated_answer}")
])

sumPrompt_prompt = ChatPromptTemplate.from_messages([
    ("system", """Your task is only to summarize the Generated Answer . 
     You will be given an Arabic question a generated answer from an AI model.
     Please summarize the generated answer in a concise manner, focusing on the key points and main ideas, NOT TALK TOO MUCH."""),

    ("human", "Data:\nQuestion: {question}\n\nGenerated Answer:\n{generated_answer}")
])

# Create the prompt template
prompt = eval_prompt
prompt2 = sumPrompt_prompt


def creatingllm(model_name,temp=0.4):
    llm = ChatOpenAI(
    model_name=evaluatedModels[0],  # or qwen-Turpo, qwen-plus, qwen-max,deepseek-r1, qwen3-8b
    openai_api_key=dashscope_api_key,
    openai_api_base=dashscope_base_url,
    temperature=temp,
    extra_body={"enable_thinking": False} )
    return llm

def response (llm, question, reference_answer, generated_answer):
    """Function to evaluate a single response."""
    messages = prompt.format_messages(
        question=question, 
        reference_answer=reference_answer, 
        generated_answer=generated_answer
    )
    response = llm.invoke(messages)
    evaluated = parser.parse(response)
    return evaluated

# Initialize the LLM
llm = creatingllm("qwen3-8b") 

# Output parser
parser = StrOutputParser()

evaluatedModels = ["Qwen-plus","qwen3-8b","deepseek-r1"]
# Evaluate row-by-row
for evaluatedModel in evaluatedModels:
    results = []    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {evaluatedModel}"):
        question = row["question"]
        reference_answer = row["answer"]
        generated_answer = row[evaluatedModel]
        
        try:
            # Create the full input to LLM
            messages = prompt.format_messages(
                question=question, 
                reference_answer=reference_answer, 
                generated_answer=generated_answer
            )
            
            response = llm.invoke(messages)
        
            evaluated = response(llm,question, reference_answer, generated_answer) 
            results.append({
                "question": question,
                evaluatedModel: evaluated.content,
            })

            print(f"Evaluated {idx + 1}/{len(df)}")
        
        except Exception as e:
            print(f" Error at row {idx}: {e}")
            results.append({
                "question": question})

    # Save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(f".\evaluated_fiqh_responses{evaluatedModel}.csv", index=False, encoding="utf-8-sig")

    countNan = 0
    totalscore = 0
    for i in range(len(output_df)):
        try :
            # Extract the score using regex
            score = re.search(r'\d+', output_df[evaluatedModel][i])
            totalscore += int(score.group()) 
        except Exception as e:
            countNan += 1
            continue

    print("Results for model:", evaluatedModel)        
    finalScore = totalscore/ ((len(output_df) - countNan)*5)  
    print(finalScore*100, "%")
    print("Total NaN scores:", countNan)
    print("Evaluation completed and saved.")

    


