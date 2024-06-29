import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
st.title("Table-Based Question Answering with Integrated Plots")
st.write("Upload a CSV or Excel file, and ask a question about the data.")

# File upload
uploaded_file = st.file_uploader("Choose a file...", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(df)  # Show full data preview

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

    # Dividing page into two halves for plots
    col1, col2 = st.columns(2)

    # Left half for plots
    with col1:
        st.subheader("Left Half Plots")

        # Line Plot
        st.subheader('Line Plot')
        x_column_line = st.selectbox("Select X-axis column for Line Plot:", df.columns)
        y_column_line = st.selectbox("Select Y-axis column for Line Plot:", df.columns)
        fig_line, ax_line = plt.subplots()
        ax_line.plot(df[x_column_line], df[y_column_line])
        ax_line.set_title('Line Plot')
        ax_line.set_xlabel(x_column_line)
        ax_line.set_ylabel(y_column_line)
        st.pyplot(fig_line)

        # Scatter Plot
        st.subheader('Scatter Plot')
        x_column_scatter = st.selectbox("Select X-axis column for Scatter Plot:", df.columns)
        y_column_scatter = st.selectbox("Select Y-axis column for Scatter Plot:", df.columns)
        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(df[x_column_scatter], df[y_column_scatter], color='g')
        ax_scatter.set_title('Scatter Plot')
        ax_scatter.set_xlabel(x_column_scatter)
        ax_scatter.set_ylabel(y_column_scatter)
        st.pyplot(fig_scatter)

        # Histogram
        st.subheader('Histogram')
        column_hist = st.selectbox("Select column for Histogram:", df.columns)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(df[column_hist], bins=20, color='c', alpha=0.75)
        ax_hist.set_title('Histogram')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Frequency')
        st.pyplot(fig_hist)

    # Right half for plots
    with col2:
        st.subheader("Right Half Plots")

        # Bar Chart
        st.subheader('Bar Chart')
        x_column_bar = st.selectbox("Select X-axis column for Bar Chart:", df.columns)
        y_column_bar = st.selectbox("Select Y-axis column for Bar Chart:", df.columns)
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(df[x_column_bar], df[y_column_bar], color='m')
        ax_bar.set_title('Bar Chart')
        ax_bar.set_xlabel(x_column_bar)
        ax_bar.set_ylabel(y_column_bar)
        st.pyplot(fig_bar)

        # Box Plot
        st.subheader('Box Plot')
        column_box = st.selectbox("Select column for Box Plot:", df.columns)
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=df[column_box], ax=ax_box)
        ax_box.set_title('Box Plot')
        st.pyplot(fig_box)

        # Heatmap
        st.subheader('Heatmap')
        fig_heatmap, ax_heatmap = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_heatmap)
        ax_heatmap.set_title('Heatmap')
        st.pyplot(fig_heatmap)

        # Pie Chart
        st.subheader('Pie Chart')
        column_pie = st.selectbox("Select column for Pie Chart:", df.columns)
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(df[column_pie].value_counts(), labels=df[column_pie].unique(), autopct='%1.1f%%')
        ax_pie.set_title('Pie Chart')
        st.pyplot(fig_pie)

        # 3D Scatter Plot
        st.subheader('3D Scatter Plot')
        x_column_3d = st.selectbox("Select X-axis column for 3D Scatter Plot:", df.columns)
        y_column_3d = st.selectbox("Select Y-axis column for 3D Scatter Plot:", df.columns)
        z_column_3d = st.selectbox("Select Z-axis column for 3D Scatter Plot:", df.columns)
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.scatter(df[x_column_3d], df[y_column_3d], df[z_column_3d], c='r', marker='o')
        ax_3d.set_title('3D Scatter Plot')
        ax_3d.set_xlabel(x_column_3d)
        ax_3d.set_ylabel(y_column_3d)
        ax_3d.set_zlabel(z_column_3d)
        st.pyplot(fig_3d)
