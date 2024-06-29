import streamlit as st
import pandas as pd
import requests

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/microsoft/tapex-base"
headers = {"Authorization": "Bearer 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"}  # Replace with your actual token

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def convert_df_to_dict(df):
    table = {}
    for column in df.columns:
        table[column] = df[column].astype(str).tolist()  # Convert all values to strings
    return table

# Streamlit app
st.title("Table-Based Question Answering")

st.write("Upload a CSV or Excel file, and ask a question about the data.")

# File upload
uploaded_file = st.file_uploader("Choose a file...", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(df.head())

    # Convert DataFrame to dictionary format
    table_dict = convert_df_to_dict(df)

    # Input query
    query_text = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if query_text:
            # Query the model
            output = query({
                "inputs": {
                    "query": query_text,
                    "table": table_dict
                },
            })
            st.write("Answer:", output)
        else:
            st.write("Please enter a question.")
