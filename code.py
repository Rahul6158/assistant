import streamlit as st
from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="Alexander-Learn/bert-finetuned-squad")

# Function to get the answer for a given question
def get_answer(question, context_file):
    with open(context_file, 'r', encoding='utf-8') as file:
        context = file.read()
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app
def main():
    st.title("Question Answering Bot")
    st.write("Enter your question:")
    question = st.text_input("Question")
    context_file = 'your_context_file.txt'  # Provide the path to your context file
    if st.button("Get Answer"):
        answer = get_answer(question, context_file)
        st.write("Answer: " + answer)

if __name__ == "__main__":
    main()
