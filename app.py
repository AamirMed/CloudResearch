import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import base64
import json
import re
import random
import string
from groq import Groq
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image
import fitz  
import io
import plotly.express as px

# --- 1. UI SETUP & GLOBAL CONFIG ---
st.set_page_config(page_title="CloudResearch", layout="wide", page_icon="⬡")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ============================================================
# ENTERPRISE SaaS UI — FULL DARK THEME OVERRIDE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── GLOBAL RESET ─── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Instrument Sans', sans-serif !important;
    background-color: #080C14 !important;
    color: #C8D6F0 !important;
}

/* ─── BACKGROUND GRID TEXTURE ─── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(59, 130, 246, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59, 130, 246, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ─── MAIN CONTENT AREA ─── */
.main .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 1400px !important;
    position: relative;
    z-index: 1;
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: #0C111D !important;
    border-right: 1px solid #1A2540 !important;
}

[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] label {
    color: #7A90B8 !important;
    font-size: 0.8rem !important;
}

/* ─── SIDEBAR SECTION HEADERS ─── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3B82F6 !important;
    margin: 0.25rem 0 0.75rem 0 !important;
    padding: 0 !important;
}

/* ─── MAIN PAGE TITLE ─── */
.stApp h1 {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #E8EDF8 !important;
    letter-spacing: -0.02em !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}

.stApp h2 {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #B8C8E8 !important;
    letter-spacing: -0.01em !important;
    padding-top: 0.5rem !important;
}

.stApp h3 {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #8A9DC0 !important;
}

/* ─── INPUTS ─── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select {
    font-family: 'Instrument Sans', sans-serif !important;
    background: #0F1624 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 6px !important;
    color: #C8D6F0 !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    outline: none !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #3A4D6A !important;
}

/* Input labels */
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stFileUploader label,
.stRadio label span {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #7A90B8 !important;
    letter-spacing: 0.01em !important;
}

/* ─── SELECTBOX ─── */
[data-baseweb="select"] > div {
    background: #0F1624 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 6px !important;
    color: #C8D6F0 !important;
    font-size: 0.85rem !important;
    font-family: 'Instrument Sans', sans-serif !important;
}

[data-baseweb="select"] > div:hover {
    border-color: #2D4A7A !important;
}

[data-baseweb="popover"] {
    background: #0F1624 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 8px !important;
}

[data-baseweb="menu"] {
    background: #0F1624 !important;
}

[data-baseweb="option"] {
    background: #0F1624 !important;
    color: #C8D6F0 !important;
    font-size: 0.85rem !important;
    font-family: 'Instrument Sans', sans-serif !important;
}

[data-baseweb="option"]:hover {
    background: #1A2540 !important;
}

/* ─── BUTTONS ─── */
.stButton button {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}

/* Primary button */
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
    background: #2563EB !important;
    border: 1px solid #3B82F6 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}

.stButton button[kind="primary"]:hover,
.stButton button[data-testid="baseButton-primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
.stButton button[kind="secondary"],
.stButton button[data-testid="baseButton-secondary"] {
    background: #0F1624 !important;
    border: 1px solid #1E2D4A !important;
    color: #8BA5D0 !important;
}

.stButton button[kind="secondary"]:hover,
.stButton button[data-testid="baseButton-secondary"]:hover {
    background: #141B2D !important;
    border-color: #2D4A7A !important;
    color: #C8D6F0 !important;
    transform: translateY(-1px) !important;
}

/* ─── FORM SUBMIT BUTTON ─── */
.stFormSubmitButton button {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    letter-spacing: 0.01em !important;
    background: #2563EB !important;
    border: 1px solid #3B82F6 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2) !important;
    transition: all 0.2s ease !important;
}

.stFormSubmitButton button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ─── DOWNLOAD BUTTON ─── */
.stDownloadButton button {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    background: #0F1624 !important;
    border: 1px solid #1E2D4A !important;
    color: #8BA5D0 !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton button:hover {
    background: #141B2D !important;
    border-color: #2D4A7A !important;
    color: #C8D6F0 !important;
}

/* ─── TABS ─── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1A2540 !important;
    gap: 0 !important;
    padding: 0 !important;
}

[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #5A7099 !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.25rem !important;
    margin-bottom: -1px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}

[data-baseweb="tab"]:hover {
    color: #A0B8D8 !important;
    background: rgba(59, 130, 246, 0.05) !important;
}

[aria-selected="true"][data-baseweb="tab"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

[data-baseweb="tab-panel"] {
    padding: 1.5rem 0 0 0 !important;
    background: transparent !important;
}

/* ─── RADIO BUTTONS ─── */
[data-testid="stRadio"] > div {
    gap: 0.5rem !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
}

[data-testid="stRadio"] label {
    background: #0F1624 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #7A90B8 !important;
}

[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(59, 130, 246, 0.1) !important;
    border-color: #3B82F6 !important;
    color: #3B82F6 !important;
}

[data-testid="stRadio"] label input {
    display: none !important;
}

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1E2D4A !important;
    border-radius: 8px !important;
    background: #0A0E1A !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #2D4A7A !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

[data-testid="stFileUploader"] button {
    background: #141B2D !important;
    border: 1px solid #1E2D4A !important;
    color: #8BA5D0 !important;
    font-size: 0.8rem !important;
    border-radius: 5px !important;
}

/* ─── ALERTS / MESSAGES ─── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    border-left-width: 3px !important;
    font-size: 0.82rem !important;
    font-family: 'Instrument Sans', sans-serif !important;
    padding: 0.65rem 1rem !important;
}

/* Info */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stInfo"] {
    background: rgba(59, 130, 246, 0.06) !important;
    border-color: #2563EB !important;
    color: #93B4E0 !important;
}

/* Success */
div[data-testid="stSuccess"] {
    background: rgba(16, 185, 129, 0.06) !important;
    border-color: #059669 !important;
    color: #6EE7B7 !important;
}

/* Error */
div[data-testid="stError"] {
    background: rgba(239, 68, 68, 0.06) !important;
    border-color: #DC2626 !important;
    color: #FCA5A5 !important;
}

/* Warning */
div[data-testid="stWarning"] {
    background: rgba(245, 158, 11, 0.06) !important;
    border-color: #D97706 !important;
    color: #FCD34D !important;
}

/* ─── EXPANDER ─── */
[data-testid="stExpander"] {
    border: 1px solid #1A2540 !important;
    border-radius: 6px !important;
    background: #0C111D !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #8BA5D0 !important;
    padding: 0.65rem 1rem !important;
    background: #0C111D !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stExpander"] summary:hover {
    color: #C8D6F0 !important;
    background: #0F1624 !important;
}

[data-testid="stExpander"] > div > div {
    padding: 0.75rem 1rem 1rem 1rem !important;
    background: #080C14 !important;
    border-top: 1px solid #1A2540 !important;
}

/* ─── DATA EDITOR / TABLE ─── */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #1A2540 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.dvn-scroller {
    background: #0A0E1A !important;
}

/* ─── DIVIDER ─── */
hr {
    border: none !important;
    border-top: 1px solid #141D30 !important;
    margin: 1.5rem 0 !important;
}

/* ─── SPINNER ─── */
[data-testid="stSpinner"] {
    color: #3B82F6 !important;
}

/* ─── TOAST ─── */
[data-testid="stToast"] {
    background: #141B2D !important;
    border: 1px solid #1E2D4A !important;
    color: #C8D6F0 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-family: 'Instrument Sans', sans-serif !important;
}

/* ─── FORM CONTAINER ─── */
[data-testid="stForm"] {
    background: #0C111D !important;
    border: 1px solid #1A2540 !important;
    border-radius: 8px !important;
    padding: 1.25rem !important;
}

/* ─── CAPTION / SMALL TEXT ─── */
.stCaption, [data-testid="stCaption"] {
    font-size: 0.75rem !important;
    color: #4A6080 !important;
    font-family: 'Instrument Sans', sans-serif !important;
}

/* ─── SUCCESS / LOGOUT BUTTON in sidebar ─── */
[data-testid="stSidebar"] .stButton button {
    font-size: 0.78rem !important;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2D4A7A; }

/* ─── PLOTLY CHARTS ─── */
.js-plotly-plot .plotly .modebar {
    background: transparent !important;
}

/* ─── SYSTEM ID BADGE STYLE via caption ─── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #141B2D !important;
    border: 1px solid #1E2D4A !important;
    color: #3B82F6 !important;
    padding: 0.1em 0.4em !important;
    border-radius: 4px !important;
}

/* ─── SIDEBAR LOGO / BRAND HEADER ─── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}

.brand-logo {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
}

.brand-name {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #E8EDF8;
    letter-spacing: -0.02em;
}

.brand-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #3B82F6;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ─── PAGE TITLE BLOCK ─── */
.page-title-block {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #141D30;
    margin-bottom: 1.5rem;
}

.page-title-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.page-title-icon {
    width: 36px;
    height: 36px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}

.page-title-text {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #E8EDF8;
    letter-spacing: -0.02em;
}

.page-title-sub {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.75rem;
    color: #4A6080;
    margin-top: 0.1rem;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
    margin-right: 0.4rem;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.status-badge {
    display: flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #10B981;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
}

/* ─── METRIC CARDS ─── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: #0C111D;
    border: 1px solid #1A2540;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s ease;
}

.metric-card:hover {
    border-color: #2D4A7A;
}

.metric-label {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4A6080;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #E8EDF8;
    letter-spacing: -0.02em;
}

.metric-sub {
    font-size: 0.7rem;
    color: #4A6080;
    margin-top: 0.2rem;
}

/* ─── SECTION DIVIDER WITH LABEL ─── */
.section-label {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3A4D6A;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #141D30;
}

/* ─── PROCESSING MODE LABEL ─── */
.mode-info {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    background: rgba(59, 130, 246, 0.05);
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 6px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 1rem;
    font-size: 0.8rem;
    color: #7A9CC8;
    font-family: 'Instrument Sans', sans-serif;
}

.mode-icon {
    color: #3B82F6;
    font-size: 0.9rem;
    margin-top: 0.05rem;
    flex-shrink: 0;
}

</style>
""", unsafe_allow_html=True)


# --- 2. HELPER FUNCTIONS ---
def get_google_sheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)

def generate_unique_id(existing_ids):
    alphabet = string.ascii_uppercase + string.digits 
    while True:
        random_str = "".join(random.choices(alphabet, k=4))
        new_id = f"CR-{random_str}"
        if new_id not in existing_ids:
            return new_id

def sync_with_google_sheets(local_dataframe, sheet_url, tab_name, mode="pull"):
    google_client = get_google_sheet_client()
    try:
        sheet = google_client.open_by_url(sheet_url).worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = google_client.open_by_url(sheet_url)
        sheet = spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="20")
    
    if mode == "pull":
        cloud_data = sheet.get_all_values()
        if len(cloud_data) > 1:
            cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
        elif len(cloud_data) == 1:
            cloud_df = pd.DataFrame(columns=cloud_data[0])
        else:
            cloud_df = pd.DataFrame()

        if not cloud_df.empty:
            cloud_df.rename(columns=lambda x: 'System_ID' if str(x).strip().lower() in ['system_id', 'system id'] else x, inplace=True)
            if 'System_ID' not in cloud_df.columns:
                existing = set()
                new_ids = []
                for _ in range(len(cloud_df)):
                    nid = generate_unique_id(existing)
                    new_ids.append(nid)
                    existing.add(nid)
                cloud_df.insert(0, 'System_ID', new_ids)
            else:
                missing_mask = cloud_df['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
                if missing_mask.any():
                    existing_valid = set(cloud_df.loc[~missing_mask, 'System_ID'].astype(str).tolist())
                    new_ids = []
                    for _ in range(missing_mask.sum()):
                        nid = generate_unique_id(existing_valid)
                        new_ids.append(nid)
                        existing_valid.add(nid)
                    cloud_df.loc[missing_mask, 'System_ID'] = new_ids
        return cloud_df

    elif mode == "push":
        # Read backup before clearing (safety net)
        backup = sheet.get_all_values()
        try:
            sheet.clear()
            if not local_dataframe.empty:
                df_to_upload = local_dataframe.copy().astype(str)
                cols = ['System_ID'] + [c for c in df_to_upload.columns if c != 'System_ID']
                df_to_upload = df_to_upload[cols]
                data_to_upload = [df_to_upload.columns.values.tolist()] + df_to_upload.values.tolist()
                sheet.update(range_name="A1", values=data_to_upload)
            else:
                data_to_upload = [local_dataframe.columns.values.tolist()]
                sheet.update(range_name="A1", values=data_to_upload)
        except Exception as e:
            if backup:
                sheet.update(range_name="A1", values=backup)
            raise e
        return local_dataframe

def compress_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((2048, 2048)) 
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()

def convert_pdf_to_images(pdf_bytes):
    image_list = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150) 
        image_list.append(pix.tobytes("jpeg"))
    return image_list

def build_final_prompt(user_prompt, abbreviations, extra_rules, anti_rules):
    abbr_text = abbreviations.strip() if abbreviations.strip() else "None"
    extra_text = extra_rules.strip() if extra_rules.strip() else "None"
    anti_text = anti_rules.strip() if anti_rules.strip() else "- Do not hallucinate values\n- Do not guess missing data"
    
    final_prompt = f"""ROLE:
You are an expert clinical data extraction system.

USER INSTRUCTION:
{user_prompt}

ABBREVIATIONS:
{abbr_text}

EXTRA RULES:
- Use clinical semantic understanding
- Expand medical abbreviations
{extra_text}

ANTI-RULES:
{anti_text}

OUTPUT REQUIREMENTS:
Return ONLY valid JSON format.
No explanations, markdown formatting, or conversational text.
Use "N/A" if missing.
"""
    return final_prompt

def blueprint_decoder(image_bytes, columns, final_prompt, model_choice):
    full_prompt = f"{final_prompt}\n\nREQUIRED COLUMNS (JSON KEYS): [{columns}]\n\nOutput a valid JSON ARRAY format: [{{...}}, {{...}}]. If the image contains multiple patients, create a separate JSON object for EACH patient. If you cannot read the image, output an empty array []."
    
    if "Gemini" in model_choice:
        compressed_bytes = compress_image(image_bytes)
        img = Image.open(io.BytesIO(compressed_bytes))
        gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(gemini_model_name)
        response = model.generate_content(
            [full_prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text.strip()
    else:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        compressed_bytes = compress_image(image_bytes)
        base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
        groq_prompt = full_prompt + " ONLY output the raw JSON array."
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[{"role": "user", "content": [{"type": "text", "text": groq_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        )
        raw_output = response.choices[0].message.content.strip()
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1]
        if "```" in raw_output:
            raw_output = raw_output.split("```")[0]
        match = re.search(r'(\[.*\]|\{.*\})', raw_output.strip(), re.DOTALL)
        return match.group(1) if match else raw_output.strip()


# --- 3. AUTHENTICATION GATEKEEPER ---
try:
    def unlock_vault(read_only_dict):
        editable_dict = {}
        for key, value in read_only_dict.items():
            if isinstance(value, dict) or hasattr(value, "items"):
                editable_dict[key] = unlock_vault(value)
            else:
                editable_dict[key] = value
        return editable_dict

    mutable_credentials = unlock_vault(st.secrets["credentials"])
    authenticator = stauth.Authenticate(
        mutable_credentials,
        st.secrets["cookie"]["name"],
        st.secrets["cookie"]["key"],
        st.secrets["cookie"]["expiry_days"]
    )
    authenticator.login()
except Exception as e:
    st.error(f"Authentication system error. Contact your administrator.")
    st.error(f"Details: {e}")
    st.stop()


# --- 4. LOGIN LOGIC ---
auth_status = st.session_state.get("authentication_status")

if auth_status == False:
    st.error("Credentials not recognised. Please try again.")

elif auth_status == None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 0 2rem 0;">
        <div style="font-family:'Instrument Sans',sans-serif;font-size:1.6rem;font-weight:700;color:#E8EDF8;letter-spacing:-0.03em;margin-bottom:0.3rem;">
            ⬡ CloudResearch
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3B82F6;letter-spacing:0.1em;text-transform:uppercase;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);padding:0.2rem 0.7rem;border-radius:3px;margin-bottom:2rem;">
            Clinical Intelligence Platform
        </div>
        <div style="font-size:0.82rem;color:#4A6080;font-family:'Instrument Sans',sans-serif;">
            Enter your credentials to access the secure workspace.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif auth_status == True:
    username = st.session_state.get("username")
    name = st.session_state.get("name")
    
    user_prefs = st.secrets["credentials"]["usernames"][username]
    saved_sheet_url = user_prefs.get("sheet_url", "")
    saved_schema = user_prefs.get("default_schema", "Age, Gender, Organism")

    if "schema_input" not in st.session_state:
        st.session_state.schema_input = saved_schema

    if "master_database" not in st.session_state:
        st.session_state.master_database = pd.DataFrame()
        if saved_sheet_url and username:
            try:
                pulled_df = sync_with_google_sheets(pd.DataFrame(), saved_sheet_url, username, mode="pull")
                st.session_state.master_database = pulled_df
                if not pulled_df.empty:
                    pulled_cols = [c for c in pulled_df.columns if c.lower() != 'system_id']
                    if pulled_cols:
                        st.session_state.schema_input = ", ".join(pulled_cols)
            except Exception as e:
                st.warning(f"Auto-sync failed on login. Pull manually before adding records. ({e})")

    # ─── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        # Brand header
        st.markdown(f"""
        <div class="brand-header">
            <div class="brand-logo">⬡</div>
            <div>
                <div class="brand-name">CloudResearch</div>
            </div>
        </div>
        <div style="margin-bottom:0.25rem;">
            <span class="brand-tag">v2.0</span>
        </div>
        """, unsafe_allow_html=True)

        # Session info
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;background:#0F1624;border:1px solid #1A2540;border-radius:6px;padding:0.6rem 0.8rem;margin:0.75rem 0 0.25rem 0;">
            <div>
                <div style="font-size:0.62rem;color:#3A4D6A;font-family:'Instrument Sans',sans-serif;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Active Session</div>
                <div style="font-size:0.82rem;color:#C8D6F0;font-family:'Instrument Sans',sans-serif;font-weight:600;">{name}</div>
            </div>
            <div><span class="status-dot"></span></div>
        </div>
        """, unsafe_allow_html=True)

        authenticator.logout("Sign Out", "sidebar")
        st.markdown('<div class="section-label">Processing Engine</div>', unsafe_allow_html=True)

        selected_model = st.selectbox(
            "Active Model",
            ["Google Gemini", "Groq (Llama 4 Vision)"],
            help="Gemini: Deep reasoning for complex documents. Groq: High-speed for structured reports."
        )

        model_is_gemini = "Gemini" in selected_model
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{'#10B981' if model_is_gemini else '#F59E0B'};background:{'rgba(16,185,129,0.06)' if model_is_gemini else 'rgba(245,158,11,0.06)'};border:1px solid {'rgba(16,185,129,0.15)' if model_is_gemini else 'rgba(245,158,11,0.15)'};padding:0.3rem 0.6rem;border-radius:4px;margin-top:0.4rem;margin-bottom:0.5rem;">
            {'◉ GEMINI-2.5-FLASH · DEEP REASONING' if model_is_gemini else '◉ LLAMA-4-SCOUT-17B · HIGH SPEED'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Database Connection</div>', unsafe_allow_html=True)
        user_sheet_url = st.text_input("Google Sheet URL", saved_sheet_url, placeholder="https://docs.google.com/...")
        project_tab = username
        st.caption(f"Isolated namespace → `{project_tab}`")

        st.markdown('<div class="section-label">Target Schema</div>', unsafe_allow_html=True)
        st.info("System IDs are auto-generated. Do not include them in schema.", icon="ℹ️")

        if "safe_schema_val" not in st.session_state:
            st.session_state.safe_schema_val = st.session_state.schema_input

        col_pull, col_push = st.columns(2)
        with col_pull:
            if st.button("↓ Pull Schema", use_container_width=True):
                if user_sheet_url and project_tab:
                    with st.spinner("Syncing..."):
                        try:
                            sheet = get_google_sheet_client().open_by_url(user_sheet_url).worksheet(project_tab)
                            cloud_headers = sheet.row_values(1)
                            if cloud_headers:
                                clean_headers = [c for c in cloud_headers if c.lower() != 'system_id']
                                st.session_state.safe_schema_val = ", ".join(clean_headers)
                                st.toast("Schema pulled.", icon="↓")
                                st.rerun()
                            else:
                                st.toast("Sheet is empty.")
                        except Exception as e:
                            st.toast(f"Error: {e}")
                else:
                    st.toast("Enter Sheet URL first.")

        with col_push:
            if st.button("↑ Push Schema", use_container_width=True):
                if user_sheet_url and project_tab:
                    with st.spinner("Syncing..."):
                        try:
                            sheet = get_google_sheet_client().open_by_url(user_sheet_url).worksheet(project_tab)
                            current_cols = [c.strip() for c in st.session_state.safe_schema_val.split(',') if c.strip()]
                            final_headers = ['System_ID'] + [c for c in current_cols if c.lower() != 'system_id']
                            sheet.update(range_name="A1", values=[final_headers])
                            st.toast("Schema pushed.", icon="↑")
                        except Exception as e:
                            st.toast(f"Error: {e}")
                else:
                    st.toast("Enter Sheet URL first.")

        updated_schema = st.text_input("Schema Columns", value=st.session_state.safe_schema_val, placeholder="Age, Gender, Organism, ALT, AST...")
        st.session_state.safe_schema_val = updated_schema
        st.session_state.schema_input = updated_schema

        st.markdown('<div class="section-label">Extraction Logic</div>', unsafe_allow_html=True)

        st.session_state.user_prompt = st.text_area(
            "Primary Directive",
            value=st.session_state.get("user_prompt", "You are an expert clinical researcher. Extract structured medical data from the provided documents."),
            height=90
        )

        with st.expander("Advanced Constraints"):
            st.session_state.abbreviations = st.text_area(
                "Abbreviation Map",
                value=st.session_state.get("abbreviations", ""),
                placeholder="DM → Diabetes Mellitus\nHTN → Hypertension\nS → Sensitive\nR → Resistant",
                height=80
            )
            st.session_state.extra_rules = st.text_area(
                "Inclusion Rules",
                value=st.session_state.get("extra_rules", ""),
                placeholder="- Prefer lab-confirmed values\n- Use latest value if multiple present",
                height=70
            )
            st.session_state.anti_rules = st.text_area(
                "Exclusion Rules",
                value=st.session_state.get("anti_rules", ""),
                placeholder="- Do not hallucinate values\n- Do not infer missing data",
                height=70
            )

        st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)
        with st.expander("⚠ Danger Zone"):
            st.warning("Clears all unsaved local data permanently.")
            if st.button("Purge Local Cache", type="primary", use_container_width=True):
                st.session_state.master_database = pd.DataFrame()
                st.rerun()

    # ─── MAIN CONTENT ──────────────────────────────────────────

    # Page header
    record_count = len(st.session_state.master_database) if not st.session_state.master_database.empty else 0
    col_count = len(st.session_state.master_database.columns) - 1 if not st.session_state.master_database.empty and len(st.session_state.master_database.columns) > 1 else 0

    st.markdown(f"""
    <div class="page-title-block">
        <div class="page-title-left">
            <div class="page-title-icon">⬡</div>
            <div>
                <div class="page-title-text">CloudResearch Command Center</div>
                <div class="page-title-sub">Clinical Intelligence Platform · Secure Workspace</div>
            </div>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            SESSION ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Records in Cache</div>
            <div class="metric-value">{record_count:,}</div>
            <div class="metric-sub">local session</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Schema Fields</div>
            <div class="metric-value">{col_count}</div>
            <div class="metric-sub">active columns</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Engine</div>
            <div class="metric-value" style="font-size:1rem;padding-top:0.3rem;">{"Gemini" if model_is_gemini else "Groq"}</div>
            <div class="metric-sub">{'gemini-2.5-flash' if model_is_gemini else 'llama-4-scout-17b'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    expected_cols = [c.strip() for c in st.session_state.schema_input.split(',') if c.strip() and c.strip().lower() != 'system_id']
    tabs = st.tabs(["  Data Entry & Synchronization  ", "  Clinical Data Explorer  "])

    # ==========================================
    # TAB 1: DATA ENTRY & SYNC
    # ==========================================
    with tabs[0]:
        entry_mode = st.radio(
            "Processing Mode",
            ["Single Record (Compile Pages)", "Batch Processing (Roster Extract)", "Update Existing Record"],
            index=0, horizontal=True
        )
        st.divider()

        # ── Single Record ──
        if "Single Record" in entry_mode:
            st.markdown("""
            <div class="mode-info">
                <span class="mode-icon">◈</span>
                Upload all pages belonging to one subject. The engine compiles scattered data across pages into a unified record and assigns a single System ID.
            </div>
            """, unsafe_allow_html=True)

            with st.form("add_single_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents (PNG, JPG, PDF)",
                    type=['png', 'jpg', 'jpeg', 'pdf'],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(f"⟳  Process via {selected_model.split(' ')[0]}", type="primary")

            if submitted and uploaded_files:
                final_prompt = build_final_prompt(
                    st.session_state.user_prompt,
                    st.session_state.abbreviations,
                    st.session_state.extra_rules,
                    st.session_state.anti_rules
                )
                with st.spinner("Pre-processing documents..."):
                    ready_images = []
                    for file in uploaded_files:
                        if file.name.lower().endswith('.pdf'):
                            ready_images.extend(convert_pdf_to_images(file.getvalue()))
                        else:
                            ready_images.append(file.getvalue())

                with st.spinner(f"Extracting and compiling profile across {len(ready_images)} page(s)..."):
                    master_patient_data = {col: 'N/A' for col in expected_cols}
                    for image_bytes in ready_images:
                        raw_json = blueprint_decoder(image_bytes, st.session_state.schema_input, final_prompt, selected_model)
                        try:
                            ai_data_list = json.loads(raw_json)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(ai_data_list, dict):
                            ai_data_list = [ai_data_list]
                        elif not isinstance(ai_data_list, list):
                            ai_data_list = []

                        for data_obj in ai_data_list:
                            for col in expected_cols:
                                new_val = str(data_obj.get(col, data_obj.get(f"{col}:", 'N/A'))).strip()
                                if new_val not in ['N/A', 'nan', '', 'None']:
                                    if master_patient_data[col] == 'N/A':
                                        master_patient_data[col] = new_val

                    current_batch_df = pd.DataFrame([master_patient_data])
                    existing_db_ids = set()
                    if not st.session_state.master_database.empty and 'System_ID' in st.session_state.master_database.columns:
                        existing_db_ids = set(st.session_state.master_database['System_ID'].astype(str).tolist())

                    new_id = generate_unique_id(existing_db_ids)
                    current_batch_df.insert(0, "System_ID", [new_id])

                    if not st.session_state.master_database.empty:
                        st.session_state.master_database = pd.concat([st.session_state.master_database, current_batch_df], ignore_index=True)
                    else:
                        st.session_state.master_database = current_batch_df

                    st.success(f"Record compiled successfully. System ID assigned: **{new_id}**")

        # ── Batch Processing ──
        elif "Batch Processing" in entry_mode:
            st.markdown("""
            <div class="mode-info">
                <span class="mode-icon">◈</span>
                Upload rosters or multi-subject documents. The engine isolates individual records and assigns a unique System ID to each subject found.
            </div>
            """, unsafe_allow_html=True)

            with st.form("add_multiple_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents (PNG, JPG, PDF)",
                    type=['png', 'jpg', 'jpeg', 'pdf'],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(f"⟳  Process Batch via {selected_model.split(' ')[0]}", type="primary")

            if submitted and uploaded_files:
                final_prompt = build_final_prompt(
                    st.session_state.user_prompt,
                    st.session_state.abbreviations,
                    st.session_state.extra_rules,
                    st.session_state.anti_rules
                )
                with st.spinner("Pre-processing documents..."):
                    ready_images = []
                    for file in uploaded_files:
                        if file.name.lower().endswith('.pdf'):
                            ready_images.extend(convert_pdf_to_images(file.getvalue()))
                        else:
                            ready_images.append(file.getvalue())

                patient_dfs = []
                for i, image_bytes in enumerate(ready_images):
                    with st.spinner(f"Analyzing page {i+1} of {len(ready_images)}..."):
                        roster_prompt = final_prompt + "\nCRITICAL: Extract EVERY subject as a separate JSON object in the array."
                        raw_json = blueprint_decoder(image_bytes, st.session_state.schema_input, roster_prompt, selected_model)
                        try:
                            ai_data_list = json.loads(raw_json)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(ai_data_list, dict):
                            ai_data_list = [ai_data_list]

                        for patient_data in ai_data_list:
                            filtered_data = {col: str(patient_data.get(col, patient_data.get(f"{col}:", 'N/A'))).strip() for col in expected_cols}
                            if any(val not in ['N/A', 'nan', '', 'None'] for val in filtered_data.values()):
                                patient_dfs.append(pd.DataFrame([filtered_data]))

                if patient_dfs:
                    current_batch_df = pd.concat(patient_dfs, ignore_index=True)
                    existing_db_ids = set()
                    if not st.session_state.master_database.empty and 'System_ID' in st.session_state.master_database.columns:
                        existing_db_ids = set(st.session_state.master_database['System_ID'].astype(str).tolist())

                    new_ids = []
                    for _ in range(len(current_batch_df)):
                        nid = generate_unique_id(existing_db_ids)
                        new_ids.append(nid)
                        existing_db_ids.add(nid)

                    current_batch_df.insert(0, "System_ID", new_ids)

                    if not st.session_state.master_database.empty:
                        st.session_state.master_database = pd.concat([st.session_state.master_database, current_batch_df], ignore_index=True)
                    else:
                        st.session_state.master_database = current_batch_df

                    st.success(f"Batch complete. **{len(current_batch_df)}** records extracted and staged.")

        # ── Update Existing ──
        elif "Update Existing" in entry_mode:
            st.markdown("""
            <div class="mode-info">
                <span class="mode-icon">◈</span>
                Append new documentation to an existing record. The engine will only fill empty fields — it will not overwrite existing values.
            </div>
            """, unsafe_allow_html=True)

            with st.form("update_form", clear_on_submit=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_id = st.text_input("System ID Reference", placeholder="CR-XXXX")
                with col2:
                    update_files = st.file_uploader(
                        "Upload Appendix Documents",
                        type=['png', 'jpg', 'jpeg', 'pdf'],
                        accept_multiple_files=True
                    )
                update_submitted = st.form_submit_button(f"⟳  Update Record via {selected_model.split(' ')[0]}", type="primary")

            if update_submitted:
                if not target_id:
                    st.error("System ID reference is required.")
                elif st.session_state.master_database.empty or target_id not in st.session_state.master_database['System_ID'].values:
                    st.error(f"System ID `{target_id}` not found in local cache. Pull from cloud first.")
                elif not update_files:
                    st.error("No documents uploaded.")
                else:
                    final_prompt = build_final_prompt(
                        st.session_state.user_prompt,
                        st.session_state.abbreviations,
                        st.session_state.extra_rules,
                        st.session_state.anti_rules
                    )
                    with st.spinner("Pre-processing documents..."):
                        ready_images = []
                        for file in update_files:
                            if file.name.lower().endswith('.pdf'):
                                ready_images.extend(convert_pdf_to_images(file.getvalue()))
                            else:
                                ready_images.append(file.getvalue())

                    for i, image_bytes in enumerate(ready_images):
                        with st.spinner(f"Extracting updates from page {i+1}..."):
                            raw_json = blueprint_decoder(image_bytes, st.session_state.schema_input, final_prompt, selected_model)
                            try:
                                ai_data = json.loads(raw_json)
                            except json.JSONDecodeError:
                                continue

                            if isinstance(ai_data, list) and len(ai_data) > 0:
                                ai_data = ai_data[0]
                            elif isinstance(ai_data, dict):
                                pass
                            else:
                                ai_data = {}

                            idx = st.session_state.master_database.index[st.session_state.master_database['System_ID'] == target_id].tolist()[0]
                            for col in expected_cols:
                                new_val = str(ai_data.get(col, 'N/A')).strip()
                                if new_val not in ['N/A', 'nan', '', 'None']:
                                    st.session_state.master_database.at[idx, col] = new_val

                    st.success(f"Record `{target_id}` updated successfully.")

        # ── Data Verification Table ──
        if not st.session_state.master_database.empty:
            st.divider()
            st.markdown('<div class="section-label">Data Verification Table</div>', unsafe_allow_html=True)
            st.caption("Review and correct AI-extracted values before committing to cloud. Double-click any cell to edit.")
            st.session_state.master_database = st.data_editor(
                st.session_state.master_database,
                num_rows="dynamic",
                use_container_width=True,
                key="data_verifier"
            )

        # ── Cloud Sync ──
        st.divider()
        st.markdown('<div class="section-label">Cloud Synchronization</div>', unsafe_allow_html=True)
        col_x, col_y, col_z = st.columns([1, 1, 1])

        with col_x:
            if st.button("↑  Commit to Cloud", type="primary", use_container_width=True):
                if st.session_state.master_database.empty:
                    st.warning("Local cache is empty. Nothing to commit.")
                else:
                    missing_ids = st.session_state.master_database['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
                    if missing_ids.any():
                        st.error(f"Validation failed: {missing_ids.sum()} record(s) missing System ID.")
                    elif not user_sheet_url or not project_tab:
                        st.error("Database connection parameters missing.")
                    else:
                        with st.spinner("Committing to cloud..."):
                            try:
                                final_cols = ['System_ID'] + [c for c in expected_cols if c.lower() != 'system_id']
                                for col in final_cols:
                                    if col not in st.session_state.master_database.columns:
                                        st.session_state.master_database[col] = 'N/A'
                                st.session_state.master_database = st.session_state.master_database[final_cols]
                                sync_with_google_sheets(st.session_state.master_database, user_sheet_url, project_tab, mode="push")
                                st.toast("Cloud commit successful.", icon="↑")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Commit failed: {e}")

        with col_y:
            if st.button("↓  Pull from Cloud", use_container_width=True):
                if not user_sheet_url or not project_tab:
                    st.error("Database connection parameters missing.")
                else:
                    with st.spinner("Downloading database instance..."):
                        try:
                            pulled_df = sync_with_google_sheets(pd.DataFrame(), user_sheet_url, project_tab, mode="pull")
                            st.session_state.master_database = pulled_df
                            if not pulled_df.empty:
                                pulled_cols = [c for c in pulled_df.columns if c.lower() != 'system_id']
                                if pulled_cols:
                                    st.session_state.schema_input = ", ".join(pulled_cols)
                            st.toast("Local cache synchronized.", icon="↓")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Sync failed: {e}")

        with col_z:
            if not st.session_state.master_database.empty:
                csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇  Export CSV",
                    data=csv_data,
                    file_name=f"{project_tab}_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # ==========================================
    # TAB 2: CLINICAL DATA EXPLORER
    # ==========================================
    with tabs[1]:
        if st.session_state.master_database.empty:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;padding:3rem 0;color:#3A4D6A;font-family:'Instrument Sans',sans-serif;">
                <div style="font-size:2rem;margin-bottom:0.75rem;">◫</div>
                <div style="font-size:0.9rem;font-weight:600;color:#4A6080;margin-bottom:0.4rem;">No data in local cache</div>
                <div style="font-size:0.78rem;">Sync from cloud or process records to enable analytics.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            df = st.session_state.master_database.copy()

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip().str.upper()
                    df[col] = df[col].replace({'N/A': 'N/A', 'NAN': 'N/A', 'NONE': 'N/A', '': 'N/A'})

            st.markdown('<div class="section-label">Visualization Parameters</div>', unsafe_allow_html=True)

            all_columns = df.columns.tolist()
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                x_axis = st.selectbox("Categorical Axis (X)", all_columns)
            with col2:
                y_axis = st.selectbox("Numerical Axis (Y)", all_columns)
            with col3:
                chart_type = st.radio("Chart", ["Bar", "Pie", "Scatter"])

            # Plotly dark template to match the app theme
            plotly_template = dict(
                layout=dict(
                    paper_bgcolor="#0C111D",
                    plot_bgcolor="#080C14",
                    font=dict(family="Instrument Sans", color="#8BA5D0", size=12),
                    title=dict(font=dict(color="#C8D6F0", size=14, family="Instrument Sans")),
                    xaxis=dict(gridcolor="#141D30", linecolor="#1A2540", tickfont=dict(color="#5A7099")),
                    yaxis=dict(gridcolor="#141D30", linecolor="#1A2540", tickfont=dict(color="#5A7099")),
                    legend=dict(bgcolor="#0C111D", bordercolor="#1A2540", borderwidth=1),
                    colorway=["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]
                )
            )

            try:
                clean_chart_df = df[~df[x_axis].isin(['N/A'])]
                clean_chart_df = clean_chart_df[~clean_chart_df[y_axis].isin(['N/A'])]

                if chart_type == "Bar":
                    fig = px.bar(clean_chart_df, x=x_axis, y=y_axis, color=x_axis,
                                 title=f"{y_axis} by {x_axis}", template=plotly_template)
                elif chart_type == "Pie":
                    fig = px.pie(clean_chart_df, names=x_axis,
                                 title=f"Distribution of {x_axis}", template=plotly_template)
                else:
                    fig = px.scatter(clean_chart_df, x=x_axis, y=y_axis, color=x_axis,
                                     title=f"{y_axis} vs {x_axis}", template=plotly_template)

                fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning("Visualization failed. Ensure the Y-axis contains continuous numerical data.")
