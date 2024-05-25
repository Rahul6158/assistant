import streamlit as st
from transformers import pipeline

# Load the question-answering pipeline
pipe = pipeline("question-answering", model="Alexander-Learn/bert-finetuned-squad")

# Streamlit app title and description
st.title("Question Answering Bot")
st.write("Ask a question and get an answer!")

# Input box for the user to enter a question
question = st.text_input("Enter your question:")

# Button to submit the question
if st.button("Ask"):
    if question:
        # Use the pipeline to get the answer
        answer = pipe({"question": question, "context": "Replace this with your context text"})
        # Display the answer
        st.write("Answer:", answer["answer"])
    else:
        st.write("Please enter a question.")

