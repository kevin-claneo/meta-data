import streamlit as st
import pandas as pd
import gspread
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Configuration: Set to True if running locally, False if running on Streamlit Cloud
# IS_LOCAL = True
IS_LOCAL = False

def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(
    page_title="Topical Authority with N-grams - Kevin (Claneo)",
    page_icon=":weight_lifter:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
        'About': "This is an app for checking your topical authority! Adapted from Lee Foot's GSC-connector check out his apps: https://leefoot.co.uk"
    }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption(":point_right: Join Claneo and support exciting clients as part of the Consulting team") 
    st.caption(':bulb: Make sure to mention that *Kevin* brought this job posting to your attention')
    st.link_button("Learn More", "https://www.claneo.com/en/career/#:~:text=Consulting")
    st.title("Check the topical authority of a GSC property")
    st.divider()

# -------------
# Google Authentication Functions
# -------------

def load_config():
    """
    Loads the Google API client configuration from Streamlit secrets.
    Returns a dictionary with the client configuration for OAuth.
    """
    client_config = {
        "installed": {
            "client_id": str(st.secrets["installed"]["client_id"]),
            "client_secret": str(st.secrets["installed"]["client_secret"]),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
            "redirect_uris": (
                ["http://localhost:8501"]
                if IS_LOCAL
                else [str(st.secrets["installed"]["redirect_uris"][0])]
            ),
        }
    }
    return client_config


def init_oauth_flow(client_config):
    """
    Initialises the OAuth flow for Google API authentication using the client configuration.
    Sets the necessary scopes and returns the configured Flow object.
    """
    scopes = ["https://www.googleapis.com/auth/webmasters"]
    return Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=client_config["installed"]["redirect_uris"][0],
    )


def google_auth(client_config):
    """
    Starts the Google authentication process using OAuth.
    Generates and returns the OAuth flow and the authentication URL.
    """
    flow = init_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt="consent")
    return flow, auth_url


# Function to access the Google Sheet
def access_google_sheet(credentials, spreadsheet_name):
    gc = gspread.authorize(credentials)
    try:
        spreadsheet = gc.open(spreadsheet_name)
        # Assuming you want to access the first sheet
        worksheet = spreadsheet.sheet1
        values = worksheet.get_all_values()
        df = pd.DataFrame(values)
        st.dataframe(df)
    except Exception as e:
        st.error(f'An error occurred: {e}')


def show_google_sign_in(auth_url):
    """
    Displays the Google sign-in button and authentication URL in the Streamlit sidebar.
    """
    with st.sidebar:
        if st.button("Sign in with Google"):
            # Open the authentication URL
            st.write('Please click the link below to sign in:')
            st.markdown(f'[Google Sign-In]({auth_url})', unsafe_allow_html=True)


def main():
    """
    The main function for the Streamlit application.
    Handles the app setup, authentication, UI components, and data fetching logic.
    """
    setup_streamlit()
    client_config = load_config()
    st.session_state.auth_flow, st.session_state.auth_url = google_auth(client_config)

    query_params = st.experimental_get_query_params()
    auth_code = query_params.get("code", [None])[0]

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        show_google_sign_in(st.session_state.auth_url)
    else:
        init_session_state()
        # Access Google Sheets data
        spreadsheet_name = st.text_input("Enter the Google Sheets name:")
        if spreadsheet_name:
            access_google_sheet(st.session_state.credentials, spreadsheet_name)

        # Continue with the rest of your app logic...
        account = auth_search_console(client_config, st.session_state.credentials)
        properties = list_gsc_properties(st.session_state.credentials)

        # The rest of your app logic goes here...

if __name__ == "__main__":
    main()
