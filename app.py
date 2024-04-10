import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq
import re
import time
import advertools as adv


# Constants
ANTHROPIC_MODELS = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229','claude-3-haiku-20240307']
GROQ_MODELS = ['mixtral-8x7b-32768', 'llama2-70b-4096']
OPENAI_MODELS = ['gpt-4-1106-preview', 'gpt-3.5-turbo']
MODELS = GROQ_MODELS + ANTHROPIC_MODELS + OPENAI_MODELS
LANGUAGES = ['German', 'English', 'Spanish', 'French', 'Italian', 'Dutch', 'Polish', 'Russian', 'Turkish', 'Arabic', 'Chinese', 'Japanese', 'Korean', 'Vietnamese', 'Indonesian', 'Hindi', 'Bengali', 'Urdu', 'Malay', 'Thai', 'Burmese', 'Cambodian', 'Amharic', 'Swahili', 'Hausa', 'Yoruba', 'Igbo', 'Oromo', 'Tigrinya', 'Afar', 'Somali', 'Ethiopian', 'Tajik', 'Pashto', 'Persian', 'Uzbek', 'Kazakh', 'Kyrgyz', 'Turkmen', 'Azerbaijani', 'Armenian', 'Georgian', 'Moldovan']
MAX_TOKENS_TITLE = 16
MAX_TOKENS_META_DESCRIPTION = 44
MAX_TOKENS_H1 = 17
TEMPERATURE = 0.7

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
        menu_items={
            'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
            'About': "This is an app for accessing Google Sheets data."
        }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption("👋 Developed by [Kevin](https://www.linkedin.com/in/kirchhoff-kevin/)") 
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
    model = st.selectbox("Choose a model:", MODELS, help=f"""
    Here's a brief overview of the models available for generating content:
    
    - **{GROQ_MODELS}**: These models are free to use and offer fast response times, making them an excellent choice for users looking for quick results. However, they may not always provide the highest quality of text. Among the GROQ models, the first model in this list: **{GROQ_MODELS[0]}** is generally considered the best due to its balance of speed and quality.
    
    - **{ANTHROPIC_MODELS}**: The models from Anthropic are known for their superior text quality. However, they require an API key, which can be obtained from [Anthropic's platform](https://console.anthropic.com/settings/keys). Among the Anthropic models, the first model in this list: **{ANTHROPIC_MODELS[0]}**, is considered the best, offering the highest quality text, but is the most costly.
    
    - **{OPENAI_MODELS}**: These are the most well-known models in the industry. You can obtain an API key from [OpenAI's platform](https://platform.openai.com/api-keys). Among the OpenAI models, the first model in this list: **{OPENAI_MODELS[0]}**, is considered the best, offering the highest quality text, but is the most costly.
    
    **It's important to note that the quality and cost-effectiveness of models can vary greatly, so choose the model wisely and test before creating loads of meta data. Always consider your specific needs and budget when selecting a model.**
    
    For the most current information on which model is performing best overall, you can visit the [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) on Hugging Face. This leaderboard provides insights into the performance of various models in real-world scenarios, helping you make an informed decision.
    """)
    if model in GROQ_MODELS:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
    elif model in ANTHROPIC_MODELS:
        client = Anthropic(api_key=st.text_input('Please enter your Anthropic API Key:', type="password"))
    elif model in OPENAI_MODELS:
        client = OpenAI(api_key=st.text_input('Please enter your OpenAI API Key:', type="password"))
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
def generate_content(client, model, text, language, meta_type):
    max_tokens = MAX_TOKENS_TITLE if meta_type == 'title tag' else (MAX_TOKENS_META_DESCRIPTION if meta_type == 'meta description' else MAX_TOKENS_H1)
    prompt = f"""
    You are a specialized assistant trained to craft the optimal {meta_type} for SEO in {language}. Your task is to produce content that is:
    - Human-like
    - Unique, ensuring each title tag, meta description, and H1 are distinct, informative, concise and keyword-optimized
    - Effective for boosting user-interaction, with a focus on capturing users' interest
    - Appealing, using varying CTAs and avoiding overuse of generic phrases
    - Avoid repetitive or boilerplate titles and ensure the most important words are front-loaded
    
    You will be given a combination of Title Tag, Meta Description, H1, and target keyword. It's possible that one or more of these inputs might be 'None' or that the page doesn't exist. In such cases, ignore these inputs and create something new based on the available information.
    
    Respond only with the new {meta_type} in the form '{meta_type}'. You must not use quotation marks, squared brackets or '{meta_type}:' around your response. 
    
    Your response must be in {language} at all cost.
    
    Adapt your language style to match the tone of the text input. Do not include any notes, explanations, or additional information. Focus solely on generating the {meta_type} for the target keyword. Try to fit in the keyword as naturally as possible, especially in the title tag and H1.
    
    Your output is limited to {max_tokens} tokens. Finish the output fully within this limit.
    
    For product pages, include important product information such as the product name, model number, price, features, benefits, and availability to make your meta description and H1 tag more relevant and appealing to potential customers.
    """
    if model in ANTHROPIC_MODELS:
        try:
            response = client.messages.create(
                model=model,
                system=prompt,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            return response.content[0].text
        except Exception as e:
                print(f"Error: {e}. Retrying in 7 seconds...")
                time.sleep(7)
    else:
            try:
                response = client.chat.completions.create(
                    model=model, 
                    messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}],
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,   
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}. Retrying in 7 seconds...")
                time.sleep(7)

def clean_up_string(s):
    if not isinstance(s, str):
        return s # Return the original value if it's not a string

    # Remove the '@@' separators
    cleaned = s.replace('@@', '\n')
    # Split by newline to get individual elements
    elements = [elem.strip() for elem in cleaned.split('\n') if elem.strip()]
    return ' '.join(elements) # Joining the elements back into a single string

def analyze_urls(dataframe, client, model, language, meta_data_to_change):
    # Initialize a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Crawl URLs
    open('crawl_file.jl', 'w').close()
    adv.crawl(dataframe['url'], 'crawl_file.jl', follow_links=False)
    crawl_df = pd.read_json('crawl_file.jl', lines=True)

    # Clean up strings in the DataFrame
    columns_to_clean = ['title', 'meta_desc', 'h1', 'h2']
    for column in columns_to_clean:
        if column not in crawl_df.columns:
            crawl_df[column] = None
    crawl_df[columns_to_clean] = crawl_df[columns_to_clean].applymap(clean_up_string)
    # Merge the original DataFrame with the crawled data
    df = pd.merge(dataframe, crawl_df[["url", "title", "meta_desc", "h1", "status"]], on=["url"])
    
    results = []
    total_rows = len(df)
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        url = row['url']
        keyword = row['keyword']
        status = row['status']
        
        if status != 200:
            title = meta_description = h1 = None
        else:
            title = row['title']
            meta_description = row['meta_desc']
            h1 = row['h1']
    
        # Combine the extracted info and keyword into a single text block
        combined_text = f"Title: {title}\nMeta Description: {meta_description}\nH1: {h1}\nKeyword: {keyword}"
    
        new_title = new_meta_description = new_h1 = None
        if 'Title Tag' in meta_data_to_change:
            new_title = generate_content(client, model, combined_text, language, 'title')
        if 'Meta Description' in meta_data_to_change:
            new_meta_description = generate_content(client, model, combined_text, language, 'meta description')
        if 'H1' in meta_data_to_change:
            new_h1 = generate_content(client, model, combined_text, language, 'h1')
    
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
            st.write('If you are unsure which model you should choose, hover over the ❔ on the right to get some help.')
            # Choose a language for the Meta Data
            language = st.selectbox("Choose a language🌐 for the Meta Data:", LANGUAGES)

            # Select which meta data to change
            meta_data_to_change = st.multiselect("Select which meta data you want to change:", options=['Title Tag', 'Meta Description', 'H1'], default=['Title Tag', 'Meta Description', 'H1'])

            # Generate Meta Data
            if st.button("Generate Meta Data"):
                     
                # Analyze URLs and generate new content
                new_df = analyze_urls(df, client, model, language, meta_data_to_change)

                # Display the new DataFrame
                show_dataframe(new_df)

                # Download DataFrame as CSV
                download_dataframe(new_df)

    else:
        st.error("It looks like your text input is empty, please fill in both text fields")
        
# Run the main function
if __name__ == "__main__":
    main()
