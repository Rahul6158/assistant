from tkinter import *
from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="Alexander-Learn/bert-finetuned-squad")

# Function to get the answer for a given question
def get_answer(question, context_file):
    with open(context_file, 'r', encoding='utf-8') as file:
        context = file.read()
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Function to handle button click event
def on_send_click():
    question = entry_question.get()
    context_file = 'your_context_file.txt'
    answer = get_answer(question, context_file)
    label_answer.config(text="Answer: " + answer)

# Create the GUI window
root = Tk()
root.title("Question Answering Bot")

# Question input
label_question = Label(root, text="Enter your question:")
label_question.pack()
entry_question = Entry(root, width=50)
entry_question.pack()

# Send button
btn_send = Button(root, text="Send", command=on_send_click)
btn_send.pack()

# Answer display
label_answer = Label(root, text="")
label_answer.pack()

# Start the GUI event loop
root.mainloop()
