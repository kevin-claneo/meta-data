import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd
from openai import OpenAI
from groq import Groq

# Constants
GROQ_MODELS = ['mixtral-8x7b-32768', 'llama2-70b-4096']
OPENAI_MODELS = ['gpt-4-turbo-preview', 'gpt-3.5-turbo']
MODELS = GROQ_MODELS + OPENAI_MODELS


def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(
        page_title="Google Sheets Access with Streamlit",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
            'About': "This is an app for accessing Google Sheets data."
        }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption(":point_right: Join Claneo and support exciting clients as part of the Consulting team") 
    st.caption(':bulb: Make sure to mention that *Kevin* brought this job posting to your attention')
    st.link_button("Learn More", "https://www.claneo.com/en/career/#:~:text=Consulting")
    st.title("Access Google Sheets Data")
    st.divider()
# Function to convert text area input into a DataFrame
def text_to_df(text):
    items = text.split(',') + text.split('\n')
    items = [item.strip() for item in items if item.strip()]
    df = pd.DataFrame(items, columns=['url'])
    return df

# Function to show the DataFrame in an expandable section
def show_dataframe(df):
    with st.expander("Preview the First 100 Rows"):
        st.dataframe(df.head(100))

# Function to handle the model selection and API key input
def handle_api_keys():
    model = st.selectbox("Choose a model:", MODELS)
    if model in GROQ_MODELS:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
    elif model in OPENAI_MODELS:
        client = OpenAI(api_key=st.text_input('Please enter your OpenAI API Key', "https://platform.openai.com/api-keys"))
    return client

# Function to download the DataFrame as a CSV
def download_dataframe(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download DataFrame as CSV",
        data=csv,
        file_name='dataframe.csv',
        mime='text/csv',
    )

# Main function to run the Streamlit app
def main():
    setup_streamlit()
    # Text area for URLs
    urls_text = st.text_area("Enter URLs (separated by commas or line breaks):")
    # Text area for keywords
    keywords_text = st.text_area("Enter Keywords (separated by commas or line breaks):")

    # Convert text areas to DataFrames
    urls_df = text_to_df(urls_text)
    keywords_df = text_to_df(keywords_text)

    # Combine the DataFrames
    df = pd.merge(urls_df, keywords_df, left_index=True, right_index=True)

    # Display the DataFrame
    show_dataframe(df)
    
    # Handle API keys and model selection
    client = handle_api_keys()

    # Download DataFrame as CSV
    download_dataframe(df)

# Run the main function
if __name__ == "__main__":
    main()
