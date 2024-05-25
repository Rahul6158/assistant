import streamlit as st
from transformers import pipeline

# Load the question-answering pipeline
pipe = pipeline("question-answering", model="Alexander-Learn/bert-finetuned-squad")

# Streamlit UI
st.title("Question Answering Bot")
st.write("Ask a question and the bot will provide an answer.")

# Text input for the question
question = st.text_input("Enter your question here:")

# Button to trigger the model
if st.button("Answer"):
    if question:
        # Use the pipeline to answer the question
        answer = pipe(question=question, context="")['answer']
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question.")

