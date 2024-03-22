import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Function to load the OAuth connection details from the secrets.toml file
def load_oauth_config():
    return {
        "type": st.secrets["installed"]["type"],
        "project_id": st.secrets["installed"]["project_id"],
        "private_key_id": st.secrets["installed"]["private_key_id"],
        "private_key": st.secrets["installed"]["private_key"],
        "client_email": st.secrets["installed"]["client_email"],
        "client_id": st.secrets["installed"]["client_id"],
        "auth_uri": st.secrets["installed"]["auth_uri"],
        "token_uri": st.secrets["installed"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["installed"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["installed"]["client_x509_cert_url"],
    }

# Function to authenticate and access the Google Sheet
def access_google_sheet(spreadsheet_url, oauth_config):
    # Extract the spreadsheet ID from the URL
    spreadsheet_id = spreadsheet_url.split("/")[-2]
    
    # Use the OAuth credentials to authenticate
    credentials = service_account.Credentials.from_service_account_info(oauth_config)
    try:
        # Build the service
        service = build('sheets', 'v4', credentials=credentials)
        # Specify the range of cells to access
        range_ = 'Sheet1!A1:Z1000' # Adjust the range as needed
        # Use the Sheets API to get the data
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_).execute()
        values = result.get('values', [])
        if not values:
            st.error('No data found.')
        else:
            # Convert the data to a DataFrame
            df = pd.DataFrame(values)
            # Display the DataFrame
            st.dataframe(df)
    except HttpError as error:
        st.error(f'An error occurred: {error}')

# Load the OAuth connection details
oauth_config = load_oauth_config()

# Prompt the user for the Google Sheets URL
spreadsheet_url = st.text_input("Enter the Google Sheets URL:")

# Access the Google Sheet
if spreadsheet_url:
    access_google_sheet(spreadsheet_url, oauth_config)
else:
    st.warning("Please enter a Google Sheets URL.")
