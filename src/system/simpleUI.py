import gradio as gr
from runscript import ask_question  # Change 'main' to your Python file name (without .py)

def qa_system(question):
    answer = ask_question(question)
    return answer

# Arabic interface
iface = gr.Interface(
    fn=qa_system,
    inputs=gr.Textbox(label="اكتب سؤالك هنا", placeholder="ما حكم صلاة الجمعة؟", lines=2),
    outputs=gr.Markdown(),#gr.Textbox(gr.Markdown(label="الإجابة"))
    title="نظام الأسئلة والأجوبة الفقهي باستخدام RAG",
    description="اكتب سؤالك الديني وسيتولى النظام الإجابة اعتمادًا على المرجع الذي تم تدريبه عليه."
)

iface.launch()


    
