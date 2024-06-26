# Data Cleaner
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Set a background color */
        font-family: Arial, sans-serif; /* Change font family */
        padding: 1rem; /* Add padding */
    }
    .stButton > button {
        background-color: #0366d6; /* Change button background color */
        color: white; /* Change button text color */
        border-radius: 5px; /* Rounded corners */
        padding: 0.5rem 1rem; /* Add padding inside the button */
        margin-top: 1rem; /* Add top margin */
        cursor: pointer; /* Change cursor on hover */
    }
    .stTextInput > div > div > input {
        border-radius: 5px; /* Rounded text input */
        padding: 0.5rem; /* Add padding inside text input */
    }
    .stSelectbox > div > div > div > div > select {
        border-radius: 5px; /* Rounded select box */
        padding: 0.5rem; /* Add padding inside select box */
    }
    .stSelectbox > div > div > div > div {
        margin-top: 1rem; /* Add top margin for select box */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

def display_data_statistics(df):
    st.subheader("Data Statistics")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    null_counts = df.isnull().sum()
    non_null_counts = df.notnull().sum()
    
    st.write("Null values count per column:")
    st.write(null_counts)
    
    st.write("Non-null values count per column:")
    st.write(non_null_counts)

def clean_data(df):
    st.subheader("Data Cleaning Operations")
    if st.button("Delete rows with null values"):
        df_cleaned = df.dropna()
        st.write("Rows with null values have been deleted.")
        display_data_statistics(df_cleaned)
        return df_cleaned
    return df

# Streamlit app setup
st.title("Table-Based Question Answering with Integrated Plots and Data Cleaning")

# File upload
uploaded_file = st.file_uploader("Choose a file...", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(df)  # Display the complete data preview

    # Convert DataFrame to dictionary with truncation
    table_dict = convert_df_to_dict(df)

    # Display data statistics and perform cleaning
    df = clean_data(df)

    # Use st.columns() to divide into two halves vertically
    left_column, right_column = st.columns(2)

    # Left half: Integrated Plots
    with left_column:
        st.subheader("Integrated Plots")

        plot_types = ['Line Plot', 'Scatter Plot', 'Histogram', 'Bar Chart', 'Box Plot', 'Violin Plot', 'Heatmap', 'Area Plot', 'Pie Chart', '3D Scatter Plot']
        plot_type = st.selectbox("Select Plot Type:", plot_types)

        if plot_type == 'Line Plot':
            st.subheader('Line Plot')
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)
            fig_line, ax_line = plt.subplots()
            ax_line.plot(df[x_column].sort_values(ascending=False).head(10), df[y_column].sort_values(ascending=False).head(10))
            ax_line.set_title('Line Plot')
            ax_line.set_xlabel(x_column)
            ax_line.set_ylabel(y_column)
            ax_line.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_line)

        elif plot_type == 'Scatter Plot':
            st.subheader('Scatter Plot')
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)
            fig_scatter, ax_scatter = plt.subplots()
            ax_scatter.scatter(df[x_column].sort_values(ascending=False).head(10), df[y_column].sort_values(ascending=False).head(10), color='g')
            ax_scatter.set_title('Scatter Plot')
            ax_scatter.set_xlabel(x_column)
            ax_scatter.set_ylabel(y_column)
            ax_scatter.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_scatter)

        elif plot_type == 'Histogram':
            st.subheader('Histogram')
            column = st.selectbox("Select column for histogram:", df.columns)
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(df[column].sort_values(ascending=False).head(10), bins=10, color='c', alpha=0.75)
            ax_hist.set_title('Histogram')
            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Frequency')
            ax_hist.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_hist)

        elif plot_type == 'Bar Chart':
            st.subheader('Bar Chart')
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(df[x_column].sort_values(ascending=False).head(10), df[y_column].sort_values(ascending=False).head(10), color='m')
            ax_bar.set_title('Bar Chart')
            ax_bar.set_xlabel(x_column)
            ax_bar.set_ylabel(y_column)
            ax_bar.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_bar)

        elif plot_type == 'Box Plot':
            st.subheader('Box Plot')
            column = st.selectbox("Select column for box plot:", df.columns)
            fig_box, ax_box = plt.subplots()
            sns.boxplot(x=df[column].sort_values(ascending=False).head(10), ax=ax_box)
            ax_box.set_title('Box Plot')
            ax_box.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_box)

        elif plot_type == 'Violin Plot':
            st.subheader('Violin Plot')
            column = st.selectbox("Select column for violin plot:", df.columns)
            fig_violin, ax_violin = plt.subplots()
            sns.violinplot(x=df[column].sort_values(ascending=False).head(10), ax=ax_violin)
            ax_violin.set_title('Violin Plot')
            ax_violin.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_violin)

        elif plot_type == 'Heatmap':
            st.subheader('Heatmap')
            fig_heatmap, ax_heatmap = plt.subplots()
            sns.heatmap(df.head(10).corr(), annot=True, cmap='coolwarm', ax=ax_heatmap)
            ax_heatmap.set_title('Heatmap')
            ax_heatmap.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_heatmap)

        elif plot_type == 'Area Plot':
            st.subheader('Area Plot')
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)
            fig_area, ax_area = plt.subplots()
            ax_area.fill_between(df[x_column].sort_values(ascending=False).head(10), df[y_column].sort_values(ascending=False).head(10), color='y', alpha=0.5)
            ax_area.set_title('Area Plot')
            ax_area.set_xlabel(x_column)
            ax_area.set_ylabel(y_column)
            ax_area.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size
            st.pyplot(fig_area)

        elif plot_type == 'Pie Chart':
            st.subheader('Pie Chart')
            column = st.selectbox("Select column for pie chart:", df.columns)
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(df[column].sort_values(ascending=False).head(10), labels=df.head(10).index, autopct='%1.1f%%', startangle=140)
            ax_pie.set_title('Pie Chart')
            ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig_pie)

        elif plot_type == '3D Scatter Plot':
            st.subheader('3D Scatter Plot')
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)
            z_column = st.selectbox("Select Z-axis column:", df.columns)
            fig_3d = plt.figure()
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.scatter(df[x_column].sort_values(ascending=False).head(10), df[y_column].sort_values(ascending=False).head(10), df[z_column].sort_values(ascending=False).head(10), c='r', marker='o')
            ax_3d.set_xlabel(x_column)
            ax_3d.set_ylabel(y_column)
            ax_3d.set_zlabel(z_column)
            st.pyplot(fig_3d)

    # Right half: Table-Based Question Answering
    with right_column:
        st.subheader("Table-Based Question Answering")

        question = st.text_input("Ask a question about the data:")
        if st.button("Get Answer"):
            payload = {
                "inputs": {
                    "question": question,
                    "table": table_dict
                }
            }
            response = query(payload)
            st.write(f"Question: {question}")
            st.write(f"Answer: {response['answer']}")

else:
    st.write("Upload a CSV or Excel file to get started.")
