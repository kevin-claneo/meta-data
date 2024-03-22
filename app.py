import streamlit as st
import pandas as pd
import gspread
from oauth2client.client import OAuth2WebServerFlow

# Function to load the OAuth2 client configuration from Streamlit secrets
def load_oauth2_config():
    return {
        "client_id": st.secrets["installed"]["client_id"],
        "client_secret": st.secrets["installed"]["client_secret"],
        "redirect_uris": st.secrets["installed"]["redirect_uris"],
    }

# Function to start the Google authentication process
def google_auth(client_config):
    flow = OAuth2WebServerFlow(
        client_id=client_config["client_id"],
        client_secret=client_config["client_secret"],
        scope=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'],
        redirect_uri=client_config["redirect_uris"][0],
    )
    auth_url = flow.step1_get_authorize_url()
    return flow, auth_url

# Function to authenticate with the Google Sheets API
def auth_sheets(flow):
    # Assuming you have a way to get the authorization code from the user
    # This could be through a callback URL or manual input
    auth_code = st.text_input("Enter the authorization code:")
    if auth_code:
        credentials = flow.step2_exchange(auth_code)
        return credentials
    else:
        st.warning("Please enter the authorization code.")
        return None

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

# Load the OAuth2 configuration
client_config = load_oauth2_config()

# Start the Google authentication process
flow, auth_url = google_auth(client_config)
st.write(f"Please go to the following URL and enter the authorization code: {auth_url}")

# Authenticate with the Google Sheets API
credentials = auth_sheets(flow)

# Prompt the user for the Google Sheets name
spreadsheet_name = st.text_input("Enter the Google Sheets name:")

# Access the Google Sheet
if credentials and spreadsheet_name:
    access_google_sheet(credentials, spreadsheet_name)
else:
    st.warning("Please complete the authentication and provide the Google Sheets name.")

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
