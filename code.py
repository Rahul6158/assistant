import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
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

# Divide the page into two halves vertically
left_column, right_column = st.beta_columns(2)

# Left half of the page
with left_column:
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

# Right half of the page for integrated plots
with right_column:
    st.subheader("Integrated Plots")

    plot_types = ['Line Plot', 'Scatter Plot', 'Histogram', 'Bar Chart', 'Box Plot', 'Violin Plot', 'Heatmap', 'Area Plot', 'Pie Chart', '3D Scatter Plot']
    plot_type = st.selectbox("Select Plot Type:", plot_types)

    if plot_type == 'Line Plot':
        st.subheader('Line Plot')
        x_column = st.selectbox("Select X-axis column:", df.columns)
        y_column = st.selectbox("Select Y-axis column:", df.columns)
        fig_line, ax_line = plt.subplots()
        ax_line.plot(df[x_column], df[y_column])
        ax_line.set_title('Line Plot')
        ax_line.set_xlabel(x_column)
        ax_line.set_ylabel(y_column)
        st.pyplot(fig_line)

    elif plot_type == 'Scatter Plot':
        st.subheader('Scatter Plot')
        x_column = st.selectbox("Select X-axis column:", df.columns)
        y_column = st.selectbox("Select Y-axis column:", df.columns)
        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(df[x_column], df[y_column], color='g')
        ax_scatter.set_title('Scatter Plot')
        ax_scatter.set_xlabel(x_column)
        ax_scatter.set_ylabel(y_column)
        st.pyplot(fig_scatter)

    elif plot_type == 'Histogram':
        st.subheader('Histogram')
        column = st.selectbox("Select column for histogram:", df.columns)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(df[column], bins=20, color='c', alpha=0.75)
        ax_hist.set_title('Histogram')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Frequency')
        st.pyplot(fig_hist)

    elif plot_type == 'Bar Chart':
        st.subheader('Bar Chart')
        x_column = st.selectbox("Select X-axis column:", df.columns)
        y_column = st.selectbox("Select Y-axis column:", df.columns)
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(df[x_column], df[y_column], color='m')
        ax_bar.set_title('Bar Chart')
        ax_bar.set_xlabel(x_column)
        ax_bar.set_ylabel(y_column)
        st.pyplot(fig_bar)

    elif plot_type == 'Box Plot':
        st.subheader('Box Plot')
        column = st.selectbox("Select column for box plot:", df.columns)
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=df[column], ax=ax_box)
        ax_box.set_title('Box Plot')
        st.pyplot(fig_box)

    elif plot_type == 'Violin Plot':
        st.subheader('Violin Plot')
        column = st.selectbox("Select column for violin plot:", df.columns)
        fig_violin, ax_violin = plt.subplots()
        sns.violinplot(x=df[column], ax=ax_violin)
        ax_violin.set_title('Violin Plot')
        st.pyplot(fig_violin)

    elif plot_type == 'Heatmap':
        st.subheader('Heatmap')
        fig_heatmap, ax_heatmap = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_heatmap)
        ax_heatmap.set_title('Heatmap')
        st.pyplot(fig_heatmap)

    elif plot_type == 'Area Plot':
        st.subheader('Area Plot')
        x_column = st.selectbox("Select X-axis column:", df.columns)
        y_column = st.selectbox("Select Y-axis column:", df.columns)
        fig_area, ax_area = plt.subplots()
        df.plot.area(x=x_column, y=y_column, ax=ax_area)
        ax_area.set_title('Area Plot')
        ax_area.set_xlabel(x_column)
        ax_area.set_ylabel(y_column)
        st.pyplot(fig_area)

    elif plot_type == 'Pie Chart':
        st.subheader('Pie Chart')
        column = st.selectbox("Select column for pie chart:", df.columns)
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(df[column].value_counts(), labels=df[column].unique(), autopct='%1.1f%%')
        ax_pie.set_title('Pie Chart')
        st.pyplot(fig_pie)

    elif plot_type == '3D Scatter Plot':
        st.subheader('3D Scatter Plot')
        x_column = st.selectbox("Select X-axis column:", df.columns)
        y_column = st.selectbox("Select Y-axis column:", df.columns)
        z_column = st.selectbox("Select Z-axis column:", df.columns)
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.scatter(df[x_column], df[y_column], df[z_column], c='r', marker='o')
        ax_3d.set_title('3D Scatter Plot')
        ax_3d.set_xlabel(x_column)
        ax_3d.set_ylabel(y_column)
        ax_3d.set_zlabel(z_column)
        st.pyplot(fig_3d)
