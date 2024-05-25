import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Load the question-answering pipeline
tokenizer = AutoTokenizer.from_pretrained("Alexander-Learn/bert-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("Alexander-Learn/bert-finetuned-squad")
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Streamlit app title and description
st.title("Question Answering Bot")
st.write("Ask a question and get an answer!")

# Input box for the user to enter a question
question = st.text_input("Enter your question:")


# Button to submit the question
if st.button("Ask"):
    if question:
        # Use the pipeline to get the answer
        answer = pipe(question=question, context="Replace this with your context text")
        # Display the answer
        st.write("Answer:", answer["answer"])
    else:
        st.write("Please enter a question.")
