import streamlit as st
import pandas as pd
import requests

# Set up API URL and headers
API_URL = "https://api-inference.huggingface.co/models/microsoft/tapex-base"
headers = {"Authorization": "Bearer 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"}  # Replace with your actual token

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def convert_df_to_dict(df, max_rows=20, max_columns=5):
    truncated_df = df.iloc[:max_rows, :max_columns]  # Limit the number of rows and columns
    table = {}
    for column in truncated_df.columns:
        table[column] = truncated_df[column].astype(str).tolist()  # Convert all values to strings
    return table

# Streamlit app setup
st.title("Table-Based Question Answering")
st.write("Upload a CSV or Excel file, and ask a question about the data.")

# File upload
uploaded_file = st.file_uploader("Choose a file...", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(df.head())

    # Convert DataFrame to dictionary with truncation
    table_dict = convert_df_to_dict(df)

    # Question input
    query_text = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if query_text:
            output = query({
                "inputs": {
                    "query": query_text,
                    "table": table_dict
                },
                "parameters": {
                    "truncation": "only_first"
                }
            })
            st.write("Answer:", output)
        else:
            st.write("Please enter a question.")
