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

# Generate some data for plots
x = np.linspace(0, 10, 100)
y = np.sin(x)
data = np.random.randn(100, 4)
dates = pd.date_range('1/1/2000', periods=100)
values = pd.Series(np.random.randn(100), index=dates).cumsum()
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]
x_3d = np.random.standard_normal(100)
y_3d = np.random.standard_normal(100)
z_3d = np.random.standard_normal(100)

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

    # Integrated Plots
    st.subheader("Integrated Plots")
    
    # Line Plot
    st.subheader('Line Plot')
    fig_line, ax_line = plt.subplots()
    ax_line.plot(x, y)
    ax_line.set_title('Line Plot')
    ax_line.set_xlabel('X-axis')
    ax_line.set_ylabel('Y-axis')
    st.pyplot(fig_line)

    # Scatter Plot
    st.subheader('Scatter Plot')
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(x, y, color='g')
    ax_scatter.set_title('Scatter Plot')
    ax_scatter.set_xlabel('X-axis')
    ax_scatter.set_ylabel('Y-axis')
    st.pyplot(fig_scatter)

    # Histogram
    st.subheader('Histogram')
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(y, bins=20, color='c', alpha=0.75)
    ax_hist.set_title('Histogram')
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')
    st.pyplot(fig_hist)

    # Bar Chart
    st.subheader('Bar Chart')
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(labels, sizes, color='m')
    ax_bar.set_title('Bar Chart')
    ax_bar.set_xlabel('Categories')
    ax_bar.set_ylabel('Values')
    st.pyplot(fig_bar)

    # Box Plot (using Seaborn)
    st.subheader('Box Plot')
    fig_box, ax_box = plt.subplots()
    sns.boxplot(data=data, ax=ax_box)
    ax_box.set_title('Box Plot')
    st.pyplot(fig_box)

    # Violin Plot (using Seaborn)
    st.subheader('Violin Plot')
    fig_violin, ax_violin = plt.subplots()
    sns.violinplot(data=data, ax=ax_violin)
    ax_violin.set_title('Violin Plot')
    st.pyplot(fig_violin)

    # Heatmap (using Seaborn)
    st.subheader('Heatmap')
    fig_heatmap, ax_heatmap = plt.subplots()
    sns.heatmap(data, ax=ax_heatmap)
    ax_heatmap.set_title('Heatmap')
    st.pyplot(fig_heatmap)

    # Area Plot (using Pandas)
    st.subheader('Area Plot')
    fig_area, ax_area = plt.subplots()
    values.plot(kind='area', ax=ax_area)
    ax_area.set_title('Area Plot')
    ax_area.set_xlabel('Dates')
    ax_area.set_ylabel('Values')
    st.pyplot(fig_area)

    # Pie Chart
    st.subheader('Pie Chart')
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax_pie.set_title('Pie Chart')
    st.pyplot(fig_pie)

    # 3D Scatter Plot
    st.subheader('3D Scatter Plot')
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(x_3d, y_3d, z_3d, c='r', marker='o')
    ax_3d.set_title('3D Scatter Plot')
    ax_3d.set_xlabel('X-axis')
    ax_3d.set_ylabel('Y-axis')
    ax_3d.set_zlabel('Z-axis')
    st.pyplot(fig_3d)
