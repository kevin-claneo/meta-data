import streamlit as st
import pandas as pd
from openai import OpenAI
from groq import Groq
import re
import time
import advertools as adv


# Constants
GROQ_MODELS = ['mixtral-8x7b-32768', 'llama2-70b-4096']
OPENAI_MODELS = ['gpt-4-turbo-preview', 'gpt-3.5-turbo']
MODELS = GROQ_MODELS + OPENAI_MODELS
LANGUAGES = ['German', 'English', 'Spanish', 'French', 'Italian', 'Dutch', 'Polish', 'Russian', 'Turkish', 'Arabic', 'Chinese', 'Japanese', 'Korean', 'Vietnamese', 'Indonesian', 'Hindi', 'Bengali', 'Urdu', 'Malay', 'Thai', 'Burmese', 'Cambodian', 'Amharic', 'Swahili', 'Hausa', 'Yoruba', 'Igbo', 'Oromo', 'Tigrinya', 'Afar', 'Somali', 'Ethiopian', 'Tajik', 'Pashto', 'Persian', 'Uzbek', 'Kazakh', 'Kyrgyz', 'Turkmen', 'Azerbaijani', 'Armenian', 'Georgian', 'Moldovan']


# -------------
# Streamlit App Configuration
# -------------


def init_session_state():
    """
    Initializes or updates the Streamlit session state variables.
    """
    if 'confirmed_preview' not in st.session_state:
        st.session_state.confirmed_preview = False


def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(
        page_title="Meta Data Optimizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
            'About': "This is an app for accessing Google Sheets data."
        }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption("👋 Written by [Kevin](https://www.linkedin.com/in/kirchhoff-kevin/)") 
    st.title("🤖 Optimize Your SEO Meta Data with AI")
    st.write("This tool creates a new Title tag, Meta Description and H1 based on your existing ones. If you don't have any yet, the tool come up with it's own.")
    st.write("You only have to enter URLs and the matching keywords.")
    st.divider()


# -------------
# Functions
# -------------

def text_to_df(text, column_name):
    items = [item.strip() for item in re.split(r',|\n', text) if item.strip()]
    df = pd.DataFrame(items, columns=[column_name])
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
        client = OpenAI(api_key=st.text_input('Please enter your OpenAI API Key:', "https://platform.openai.com/api-keys"))
    return client, model

# Function to download the DataFrame as a CSV
def download_dataframe(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Your new Meta-Data as CSV",
        data=csv,
        file_name='meta-data.csv',
        mime='text/csv',
    )

# Function to generate content
def generate_content(client, model, text, language):
    while True:
        try:
            response = client.chat.completions.create(model=model, messages=[
                {"role": "system", "content": f"You are a specialized assistant trained to craft the optimal title, meta description, and h1 heading for SEO in {language}. Your task is to produce content that is human-like, unique, and effective for boosting Click-Through Rate (CTR). You will be given the current title, meta description, h1 heading, and target keyword. It's possible that one or more of these inputs might be 'None' or that the page doesn't exist. In such cases, ignore these inputs and create something new based on the available information. Respond in the exact format: 'Title: [your title here]\\nMeta Description: [your meta description here]\\nH1: [your h1 heading here]' without any quotation marks around the content. Your response should be in {language}. Adapt your language style to match the tone of the current meta data. Do not include any notes, explanations, or additional information. Focus solely on generating the title, meta description, and h1 heading."},
                {"role": "user", "content": text}
            ])
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}. Retrying in 7 seconds...")
            time.sleep(7)

def parse_gpt_response(response_str):
    pattern = r"Title: (\"?)(?P<title>[^\"]+)(\"?)\nMeta Description: (\"?)(?P<meta_description>[^\"]+)(\"?)\nH1: (\"?)(?P<h1>[^\"]+)(\"?)"
    match = re.search(pattern, response_str)

    if match:
        return match.group('title'), match.group('meta_description'), match.group('h1')
    else:
        return None, None, None
# Function to clean up strings
def clean_up_string(s):
    if not isinstance(s, str):
        return s # Return the original value if it's not a string

    # Remove the '@@' separators
    cleaned = s.replace('@@', '\n')
    # Split by newline to get individual elements
    elements = [elem.strip() for elem in cleaned.split('\n') if elem.strip()]
    return ' '.join(elements) # Joining the elements back into a single string

def analyze_urls(dataframe, client, model, language):
    # Initialize a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Crawl URLs
    open('crawl_file.jl', 'w').close()
    adv.crawl(dataframe['url'], 'crawl_file.jl', follow_links=False)
    crawl_df = pd.read_json('crawl_file.jl', lines=True)

    # Clean up strings in the DataFrame
    columns_to_clean = ['title', 'meta_desc', 'h1', 'h2']
    crawl_df[columns_to_clean] = crawl_df[columns_to_clean].applymap(clean_up_string)

    # Merge the original DataFrame with the crawled data
    df = pd.merge(dataframe, crawl_df[["url", "title", "meta_desc", "h1"]], on=["url"])

    results = []
    total_rows = len(df)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        url = row['url']
        keyword = row['keyword']
        title = row['title']
        meta_description = row['meta_desc']
        h1 = row['h1']

        # Combine the extracted info and keyword into a single text block
        combined_text = f"Title: {title}\nMeta Description: {meta_description}\nH1: {h1}\nKeyword: {keyword}"

        # Get new title, meta description, and h1 using the GPT API call
        generated_response = generate_content(client, model, combined_text, language)

        # Parse the GPT response
        new_title, new_meta_description, new_h1 = parse_gpt_response(generated_response)

        # Append the generated content to the results list
        results.append({
            "url": url,
            "new title": new_title,
            "new meta_desc": new_meta_description,
            "new h1": new_h1
        })

        # Update the progress bar
        progress = (index + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing row {index + 1} of {total_rows}")

    return pd.DataFrame(results)



# Main function to run the Streamlit app
def main():
    setup_streamlit()
    init_session_state() # Initialize session state

    # Text area for URLs
    urls_text = st.text_area("Enter URLs 🔗 (separated by commas or line breaks):")
    # Text area for keywords
    keywords_text = st.text_area("Enter Keywords 🔑 (separated by commas or line breaks):")

    # Convert text areas to DataFrames
    urls_df = text_to_df(urls_text, 'url')
    keywords_df = text_to_df(keywords_text, 'keyword')
    
    # Check if the number of URLs matches the number of keywords
    if len(urls_df) != len(keywords_df):
        st.error("The number of URLs does not match the number of keywords. Please ensure that each URL has a corresponding keyword.")
        return
    
    # Merge the DataFrames on the reset indices
    df = pd.merge(urls_df, keywords_df, left_index=True, right_index=True)

    if not df.empty:
        # Display the DataFrame
        st.write('Preview of the first 100 rows of your data, please make sure that it matches as intended.')
        st.dataframe(df.head(100))
        
        # Confirm the preview
        if st.button("Confirm Preview"):
            st.session_state.confirmed_preview = True

        if st.session_state.confirmed_preview:
            # Handle API keys and model selection
            client, model = handle_api_keys()

            # Choose a language for the Meta Data
            language = st.selectbox("Choose a language 🇺🇳 for the Meta Data:", LANGUAGES)

            # Generate Meta Data
            if st.button("Generate Meta Data"):
                # Analyze URLs and generate new content
                new_df = analyze_urls(df, client, model, language)

                # Display the new DataFrame
                show_dataframe(new_df)

                # Download DataFrame as CSV
                download_dataframe(new_df)
    else:
        st.write("It looks like your text_input is empty, please fill in both fields")

# Run the main function
if __name__ == "__main__":
    main()
