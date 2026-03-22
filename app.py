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
# CLOUDRESEARCH — PRECISION DARK UI SYSTEM
# Typography: IBM Plex Sans (body) + Syne (display)
# Palette: Deep navy base, electric teal accent, slate surfaces
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>

/* ── ROOT VARIABLES ───────────────────────────────────────── */
:root {
    --bg:          #07090F;
    --surface:     #0D1119;
    --panel:       #111620;
    --border:      #1E2535;
    --border-lit:  #2A3550;
    --accent:      #00D4AA;
    --accent-dim:  rgba(0, 212, 170, 0.12);
    --accent-glow: rgba(0, 212, 170, 0.25);
    --red:         #FF4B6E;
    --amber:       #FFB547;
    --blue:        #4B9FFF;
    --text-hi:     #EDF0F7;
    --text-md:     #8B95AE;
    --text-lo:     #4A5468;
    --font-sans:   'IBM Plex Sans', sans-serif;
    --font-display:'Syne', sans-serif;
    --font-mono:   'IBM Plex Mono', monospace;
    --radius:      6px;
    --radius-lg:   10px;
}

/* ── GLOBAL RESET ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg) !important;
    color: var(--text-hi) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── ANIMATED TOP STRIPE ──────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), #4B9FFF, var(--accent), transparent);
    background-size: 300% 100%;
    animation: stripe 4s linear infinite;
    z-index: 9999;
}
@keyframes stripe {
    0%   { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

/* ── SIDEBAR ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Sidebar logo header area */
[data-testid="stSidebar"] .stMarkdown h1 {
    font-family: var(--font-display) !important;
    color: var(--accent) !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    color: var(--text-md) !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── MAIN HEADINGS ────────────────────────────────────────── */
.stApp h1 {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-hi) !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
    line-height: 1.2 !important;
}

.stApp h2 {
    font-family: var(--font-display) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-md) !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    margin-bottom: 1rem !important;
    margin-top: 0.5rem !important;
}

.stApp h3 {
    font-family: var(--font-display) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
    margin-bottom: 0.5rem !important;
}

/* ── TABS ─────────────────────────────────────────────────── */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 1.5rem !important;
}

button[data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-md) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.2rem !important;
    letter-spacing: 0.03em !important;
    transition: color 0.2s !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--text-hi) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
}

[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

[data-baseweb="tab-border"] {
    background-color: var(--border) !important;
}

/* ── INPUTS & TEXT AREAS ──────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div:first-child {
    background-color: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
    outline: none !important;
}

[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-md) !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 0.3rem !important;
}

/* ── SELECTBOX ────────────────────────────────────────────── */
[data-baseweb="select"] * {
    background-color: var(--panel) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
}

[data-baseweb="popover"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius) !important;
}

[data-baseweb="menu"] li {
    font-size: 0.85rem !important;
}

[data-baseweb="menu"] li:hover {
    background-color: var(--accent-dim) !important;
}

/* ── BUTTONS ──────────────────────────────────────────────── */
[data-testid="stButton"] button {
    background-color: var(--panel) !important;
    color: var(--text-hi) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
}

[data-testid="stButton"] button:hover {
    background-color: var(--border-lit) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* Primary button */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #00C49A, #00A882) !important;
    color: #001F19 !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00D4AA, #00B990) !important;
    color: #001F19 !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Form submit button */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #00C49A, #00A882) !important;
    color: #001F19 !important;
    border: none !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem 1.2rem !important;
    text-transform: uppercase !important;
    transition: all 0.18s ease !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #00D4AA, #00B990) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background-color: transparent !important;
    color: var(--text-md) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    border-radius: var(--radius) !important;
    transition: all 0.18s !important;
}

[data-testid="stDownloadButton"] button:hover {
    border-color: var(--blue) !important;
    color: var(--blue) !important;
}

/* ── RADIO BUTTONS ────────────────────────────────────────── */
[data-testid="stRadio"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
}

[data-testid="stRadio"] [data-testid="stMarkdown"] p {
    font-size: 0.82rem !important;
}

/* Radio option container */
[data-testid="stRadio"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.18s !important;
}

[data-testid="stRadio"] > div > div:has(input:checked) {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── ALERT BOXES ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1rem !important;
}

/* Info */
div[data-testid="stAlert"][class*="info"] {
    background-color: rgba(75,159,255,0.08) !important;
    border-color: var(--blue) !important;
    color: #A0C4FF !important;
}

/* Success */
div[data-testid="stAlert"][class*="success"] {
    background-color: rgba(0,212,170,0.08) !important;
    border-color: var(--accent) !important;
    color: #80EDD9 !important;
}

/* Warning */
div[data-testid="stAlert"][class*="warning"] {
    background-color: rgba(255,181,71,0.08) !important;
    border-color: var(--amber) !important;
    color: #FFD59F !important;
}

/* Error */
div[data-testid="stAlert"][class*="error"] {
    background-color: rgba(255,75,110,0.08) !important;
    border-color: var(--red) !important;
    color: #FF9EB4 !important;
}

/* ── EXPANDER ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-md) !important;
    padding: 0.6rem 0.8rem !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--text-hi) !important;
    background: var(--panel) !important;
}

[data-testid="stExpander"] summary svg {
    fill: var(--text-lo) !important;
}

/* ── FILE UPLOADER ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--surface) !important;
    border: 1.5px dashed var(--border-lit) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

[data-testid="stFileUploader"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
}

[data-testid="stFileUploader"] button {
    font-size: 0.78rem !important;
}

/* ── DATA EDITOR / TABLE ──────────────────────────────────── */
[data-testid="stDataEditor"],
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

.glideDataEditor {
    background: var(--surface) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* ── SPINNER ──────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
}

[data-testid="stSpinner"] svg {
    color: var(--accent) !important;
}

/* ── TOAST ────────────────────────────────────────────────── */
[data-testid="stToast"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-hi) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}

/* ── DIVIDER ──────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── CAPTION & SMALL TEXT ─────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    font-family: var(--font-sans) !important;
    font-size: 0.72rem !important;
    color: var(--text-lo) !important;
    letter-spacing: 0.02em !important;
}

/* ── SIDEBAR SESSION STATUS ───────────────────────────────── */
[data-testid="stSidebar"] .stSuccess {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    border-radius: var(--radius) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 0.8rem !important;
}

/* ── FORM CONTAINER ───────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem !important;
}

/* ── SCROLLBAR ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-lo); }

/* ── METRIC CARDS ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.2rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-lo) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
}

/* ── CHECKBOX ─────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
    font-family: var(--font-sans) !important;
}

/* ── PLOTLY CHARTS ────────────────────────────────────────── */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* ── CUSTOM SECTION LABEL ─────────────────────────────────── */
.section-label {
    font-family: var(--font-sans);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-lo);
    margin-bottom: 0.75rem;
    margin-top: 0.5rem;
}

/* ── COLUMN LAYOUT TWEAKS ─────────────────────────────────── */
[data-testid="column"] {
    padding: 0 0.4rem !important;
}

/* ── MAIN CONTENT PADDING ─────────────────────────────────── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── TITLE BADGE ──────────────────────────────────────────── */
.cr-title-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.2rem;
}

.cr-badge {
    display: inline-block;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 0.4rem;
}

/* ── ID PILL STYLE ────────────────────────────────────────── */
.id-pill {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    background: var(--accent-dim);
    color: var(--accent);
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    border: 1px solid rgba(0,212,170,0.2);
    letter-spacing: 0.05em;
}

/* ── STATUS DOT ───────────────────────────────────────────── */
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    margin-right: 0.4rem;
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── SIDEBAR LOGO MARK ────────────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}

.sidebar-hexmark {
    width: 28px; height: 28px;
    background: var(--accent-dim);
    border: 1.5px solid var(--accent);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    color: var(--accent);
    font-weight: 700;
    font-family: var(--font-mono);
    flex-shrink: 0;
}

.sidebar-wordmark {
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-weight: 800;
    color: var(--text-hi);
    letter-spacing: 0.04em;
    line-height: 1;
}

.sidebar-wordmark span {
    color: var(--accent);
}

.sidebar-version {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-lo);
    letter-spacing: 0.1em;
    margin-top: 1px;
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
        # Read backup before clearing
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
    
    return f"""ROLE:
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
    st.error(f"Authentication system error. Please contact your administrator.")
    st.error(f"Technical detail: {e}")
    st.stop()


# --- 4. LOGIN LOGIC ---
auth_status = st.session_state.get("authentication_status")

if auth_status == False:
    st.markdown("""
        <div style="max-width:380px; margin:6rem auto; text-align:center;">
            <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#EDF0F7; margin-bottom:0.3rem;">
                Cloud<span style="color:#00D4AA">Research</span>
            </div>
            <div style="font-size:0.72rem; color:#4A5468; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2rem;">
                Clinical Intelligence Platform
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.error("Invalid credentials. Access denied.")

elif auth_status == None:
    st.markdown("""
        <div style="max-width:380px; margin:5rem auto; text-align:center;">
            <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#EDF0F7; margin-bottom:0.3rem;">
                Cloud<span style="color:#00D4AA">Research</span>
            </div>
            <div style="font-size:0.7rem; color:#4A5468; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:2.5rem;">
                Clinical Intelligence Platform
            </div>
            <div style="font-size:0.8rem; color:#6B7A99; line-height:1.6;">
                Enter your credentials to access the secure command center.
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
                st.warning(f"Auto-sync failed on login. Pull manually before adding records. Detail: {e}")

    # ── SIDEBAR ─────────────────────────────────────────────
    with st.sidebar:

        # Logo mark
        st.markdown(f"""
            <div class="sidebar-logo">
                <div class="sidebar-hexmark">⬡</div>
                <div>
                    <div class="sidebar-wordmark">Cloud<span>Research</span></div>
                    <div class="sidebar-version">v2.0 · CLINICAL</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Session status
        st.markdown(f"""
            <div style="background:rgba(0,212,170,0.07); border:1px solid rgba(0,212,170,0.2); border-radius:6px; 
                        padding:0.5rem 0.75rem; margin-bottom:1rem; display:flex; align-items:center; gap:0.5rem;">
                <span class="status-dot"></span>
                <span style="font-family:'IBM Plex Sans',sans-serif; font-size:0.75rem; color:#80EDD9; font-weight:500;">
                    {name}
                </span>
            </div>
        """, unsafe_allow_html=True)

        authenticator.logout("Sign Out", "sidebar")
        st.divider()

        st.header("Processing Engine")
        selected_model = st.selectbox(
            "Model Selection:",
            ["Google Gemini", "Groq (Llama 4 Vision)"],
            help="Gemini: Deep reasoning for complex documents. Groq: High-speed for structured rosters."
        )

        st.divider()

        st.header("Database Connection")
        user_sheet_url = st.text_input("Google Sheet URL:", saved_sheet_url, placeholder="https://docs.google.com/...")
        project_tab = username
        st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#4A5468; 
                        margin-top:0.3rem; display:flex; align-items:center; gap:0.4rem;">
                <span style="color:#2A3550;">⬡</span> 
                Active directory: <span style="color:#6B7A99;">{project_tab}</span>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.header("Target Schema")
        st.info("System IDs are auto-generated. Do not define them here.")

        if "safe_schema_val" not in st.session_state:
            st.session_state.safe_schema_val = st.session_state.schema_input

        col_pull, col_push = st.columns(2)
        with col_pull:
            if st.button("↓ Pull Schema", use_container_width=True):
                if user_sheet_url and project_tab:
                    with st.spinner("Synchronizing..."):
                        try:
                            sheet = get_google_sheet_client().open_by_url(user_sheet_url).worksheet(project_tab)
                            cloud_headers = sheet.row_values(1)
                            if cloud_headers:
                                clean_headers = [c for c in cloud_headers if c.lower() != 'system_id']
                                st.session_state.safe_schema_val = ", ".join(clean_headers)
                                st.toast("Schema synchronized from cloud.")
                                st.rerun()
                            else:
                                st.toast("Target sheet is empty.")
                        except Exception as e:
                            st.toast(f"Sync error: {e}")
                else:
                    st.toast("Database URL required.")

        with col_push:
            if st.button("↑ Push Schema", use_container_width=True):
                if user_sheet_url and project_tab:
                    with st.spinner("Synchronizing..."):
                        try:
                            sheet = get_google_sheet_client().open_by_url(user_sheet_url).worksheet(project_tab)
                            current_cols = [c.strip() for c in st.session_state.safe_schema_val.split(',') if c.strip()]
                            final_headers = ['System_ID'] + [c for c in current_cols if c.lower() != 'system_id']
                            sheet.update(range_name="A1", values=[final_headers])
                            st.toast("Schema committed to cloud.")
                        except Exception as e:
                            st.toast(f"Sync error: {e}")
                else:
                    st.toast("Database URL required.")

        updated_schema = st.text_input("Schema Columns:", value=st.session_state.safe_schema_val,
                                        placeholder="Age, Gender, Organism, Antibiotic...")
        st.session_state.safe_schema_val = updated_schema
        st.session_state.schema_input = updated_schema

        st.divider()
        st.header("Extraction Logic")
        st.session_state.user_prompt = st.text_area(
            "Primary Directive:",
            value=st.session_state.get("user_prompt", "You are an expert clinical researcher. Extract structured medical data from the provided documents."),
            height=80
        )

        with st.expander("Advanced Extraction Constraints"):
            st.session_state.abbreviations = st.text_area(
                "Abbreviations Map:",
                value=st.session_state.get("abbreviations", ""),
                placeholder="DM → Diabetes Mellitus\nHTN → Hypertension\nS → Sensitive\nR → Resistant",
                height=90
            )
            st.session_state.extra_rules = st.text_area(
                "Inclusion Rules:",
                value=st.session_state.get("extra_rules", ""),
                placeholder="- Prefer lab-confirmed values\n- Use latest value if multiple",
                height=80
            )
            st.session_state.anti_rules = st.text_area(
                "Exclusion Rules:",
                value=st.session_state.get("anti_rules", ""),
                placeholder="- Do not hallucinate values\n- Do not infer missing data",
                height=80
            )

        st.divider()
        st.header("System Controls")
        with st.expander("⚠ Danger Zone"):
            st.warning("Clears all unsaved local data permanently.")
            if st.button("Purge Local Cache", type="primary", use_container_width=True):
                st.session_state.master_database = pd.DataFrame()
                st.rerun()


    # ── MAIN HEADER ─────────────────────────────────────────
    record_count = len(st.session_state.master_database) if not st.session_state.master_database.empty else 0
    model_short = selected_model.split(" ")[0]

    st.markdown(f"""
        <div style="display:flex; align-items:flex-end; justify-content:space-between; 
                    padding-bottom:1.2rem; border-bottom:1px solid #1E2535; margin-bottom:1.5rem;">
            <div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#4A5468; 
                            letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.4rem;">
                    ⬡ CLOUDRESEARCH · COMMAND CENTER
                </div>
                <h1 style="font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800; 
                           color:#EDF0F7; margin:0; letter-spacing:-0.02em; line-height:1.1;">
                    Clinical Data<br><span style="color:#00D4AA;">Intelligence Platform</span>
                </h1>
            </div>
            <div style="display:flex; gap:1rem; align-items:center;">
                <div style="text-align:center; padding:0.6rem 1.1rem; background:#0D1119; 
                            border:1px solid #1E2535; border-radius:8px;">
                    <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#EDF0F7;">{record_count}</div>
                    <div style="font-size:0.62rem; color:#4A5468; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">Records</div>
                </div>
                <div style="text-align:center; padding:0.6rem 1.1rem; background:rgba(0,212,170,0.07); 
                            border:1px solid rgba(0,212,170,0.2); border-radius:8px;">
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:500; color:#00D4AA;">{model_short}</div>
                    <div style="font-size:0.62rem; color:#4A5468; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">Engine</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["  Data Entry & Synchronization  ", "  Clinical Data Explorer  "])
    expected_cols = [c.strip() for c in st.session_state.schema_input.split(',') if c.strip() and c.strip().lower() != 'system_id']


    # ==========================================
    # TAB 1: DATA ENTRY & SYNC
    # ==========================================
    with tabs[0]:
        st.subheader("Record Management")

        entry_mode = st.radio(
            "Processing Mode:",
            ["Single Record (Compile Pages)", "Batch Processing (Roster Extract)", "Update Existing Record"],
            index=0, horizontal=True
        )
        st.divider()

        if "Single Record" in entry_mode:
            st.info("Upload all pages for a single subject. The engine will compile scattered data into a unified profile and assign one System ID.")
            with st.form("add_single_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents (PNG, JPG, PDF):",
                    type=['png', 'jpg', 'jpeg', 'pdf'],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(f"▶  Process via {model_short}", type="primary")

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

                with st.spinner(f"Extracting parameters across {len(ready_images)} page(s)..."):
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
                                new_val = str(data_obj.get(col, data_obj.get(f"{col}:", 'N/A'))).strip().upper()
                                if new_val not in ['N/A', 'NAN', '', 'NONE']:
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

                    st.success(f"Record compiled successfully. Assigned System ID: **{new_id}**")

        elif "Batch Processing" in entry_mode:
            st.info("Upload rosters or multi-subject documents. The engine will isolate each entity and assign unique System IDs automatically.")
            with st.form("add_multiple_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Roster Documents (PNG, JPG, PDF):",
                    type=['png', 'jpg', 'jpeg', 'pdf'],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(f"▶  Process Batch via {model_short}", type="primary")

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
                        roster_prompt = final_prompt + "\nCRITICAL: Extract EVERY subject as a separate JSON object."
                        raw_json = blueprint_decoder(image_bytes, st.session_state.schema_input, roster_prompt, selected_model)
                        try:
                            ai_data_list = json.loads(raw_json)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(ai_data_list, dict):
                            ai_data_list = [ai_data_list]

                        for patient_data in ai_data_list:
                            filtered_data = {
                                col: str(patient_data.get(col, patient_data.get(f"{col}:", 'N/A'))).strip().upper()
                                for col in expected_cols
                            }
                            if any(val not in ['N/A', 'NAN', '', 'NONE'] for val in filtered_data.values()):
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

                    st.success(f"Batch complete. **{len(current_batch_df)}** records extracted and queued for verification.")

        elif "Update Existing" in entry_mode:
            st.info("Append new documentation to an existing record. The engine will only overwrite empty fields.")
            with st.form("update_form", clear_on_submit=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_id = st.text_input("System ID Reference:", placeholder="CR-XXXX")
                with col2:
                    update_files = st.file_uploader(
                        "Upload Appendices (PNG, JPG, PDF):",
                        type=['png', 'jpg', 'jpeg', 'pdf'],
                        accept_multiple_files=True
                    )
                update_submitted = st.form_submit_button(f"▶  Update Record via {model_short}", type="primary")

            if update_submitted:
                if not target_id:
                    st.error("System ID reference is required.")
                elif st.session_state.master_database.empty or target_id not in st.session_state.master_database['System_ID'].values:
                    st.error(f"System ID **{target_id}** not found in local cache. Pull from cloud first.")
                elif not update_files:
                    st.error("No documentation uploaded.")
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

                            idx = st.session_state.master_database.index[
                                st.session_state.master_database['System_ID'] == target_id
                            ].tolist()[0]
                            for col in expected_cols:
                                new_val = str(ai_data.get(col, 'N/A')).strip().upper()
                                if new_val not in ['N/A', 'NAN', '', 'NONE']:
                                    st.session_state.master_database.at[idx, col] = new_val
                    st.success(f"Record **{target_id}** updated successfully.")

        # ── VERIFICATION TABLE ───────────────────────────────
        if not st.session_state.master_database.empty:
            st.divider()
            st.subheader("Data Verification Interface")
            st.markdown("""
                <p style="font-size:0.75rem; color:#4A5468; margin-bottom:0.8rem; font-family:'IBM Plex Sans',sans-serif;">
                    Double-click any cell to correct values before committing to cloud. 
                    All changes are held in local memory until explicitly committed.
                </p>
            """, unsafe_allow_html=True)
            st.session_state.master_database = st.data_editor(
                st.session_state.master_database,
                num_rows="dynamic",
                use_container_width=True,
                key="data_verifier"
            )

        # ── CLOUD SYNC ───────────────────────────────────────
        st.divider()
        st.subheader("Cloud Synchronization")

        col_x, col_y, col_z = st.columns([1, 1, 1])

        with col_x:
            if st.button("↑  Commit to Cloud", type="primary", use_container_width=True):
                if st.session_state.master_database.empty:
                    st.warning("Local cache is empty. Nothing to commit.")
                else:
                    missing_ids = st.session_state.master_database['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
                    if missing_ids.any():
                        st.error(f"Validation error: {missing_ids.sum()} record(s) are missing a System ID.")
                    elif not user_sheet_url or not project_tab:
                        st.error("Database connection parameters are missing.")
                    else:
                        with st.spinner("Committing to Google servers..."):
                            try:
                                final_cols = ['System_ID'] + [c for c in expected_cols if c.lower() != 'system_id']
                                for col in final_cols:
                                    if col not in st.session_state.master_database.columns:
                                        st.session_state.master_database[col] = 'N/A'
                                st.session_state.master_database = st.session_state.master_database[final_cols]
                                sync_with_google_sheets(st.session_state.master_database, user_sheet_url, project_tab, mode="push")
                                st.toast("Cloud commit successful.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Commit failed: {e}")

        with col_y:
            if st.button("↓  Pull from Cloud", use_container_width=True):
                with st.spinner("Downloading database instance..."):
                    if not user_sheet_url or not project_tab:
                        st.error("Database connection parameters are missing.")
                    else:
                        try:
                            pulled_df = sync_with_google_sheets(pd.DataFrame(), user_sheet_url, project_tab, mode="pull")
                            st.session_state.master_database = pulled_df
                            if not pulled_df.empty:
                                pulled_cols = [c for c in pulled_df.columns if c.lower() != 'system_id']
                                if pulled_cols:
                                    st.session_state.schema_input = ", ".join(pulled_cols)
                            st.toast("Local cache synchronized with cloud.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Sync failed: {e}")

        with col_z:
            if not st.session_state.master_database.empty:
                csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "↓  Export Local CSV",
                    data=csv_data,
                    file_name=f"{project_tab}_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )


    # ==========================================
    # TAB 2: CLINICAL DATA EXPLORER
    # ==========================================
    with tabs[1]:
        st.subheader("Clinical Data Explorer")

        if st.session_state.master_database.empty:
            st.markdown("""
                <div style="text-align:center; padding:4rem 2rem; border:1px dashed #1E2535; border-radius:10px; margin-top:1rem;">
                    <div style="font-size:1.8rem; margin-bottom:1rem; opacity:0.3;">⬡</div>
                    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem; color:#4A5468;">
                        No data available. Synchronize with cloud or process records to enable analytics.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            df = st.session_state.master_database.copy()

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip().str.upper()
                    df[col] = df[col].replace({'N/A': 'N/A', 'NAN': 'N/A', 'NONE': 'N/A', '': 'N/A'})

            # Summary metrics row
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            total_records = len(df)
            complete_records = df.dropna().shape[0]
            schema_cols = len(df.columns) - 1

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Records", total_records)
            with m2:
                st.metric("Schema Columns", schema_cols)
            with m3:
                st.metric("Numeric Fields", len(numeric_cols))

            st.divider()

            all_columns = df.columns.tolist()
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                x_axis = st.selectbox("Categorical Axis (X)", all_columns)
            with col2:
                y_axis = st.selectbox("Numerical Axis (Y)", all_columns)
            with col3:
                chart_type = st.selectbox("Chart Type", ["Bar", "Pie", "Scatter"])

            # Plotly dark theme to match app
            plot_template = {
                "layout": {
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "#0D1119",
                    "font": {"family": "IBM Plex Sans, sans-serif", "color": "#8B95AE", "size": 11},
                    "title": {"font": {"family": "Syne, sans-serif", "color": "#EDF0F7", "size": 14}},
                    "xaxis": {"gridcolor": "#1E2535", "linecolor": "#1E2535", "tickcolor": "#4A5468"},
                    "yaxis": {"gridcolor": "#1E2535", "linecolor": "#1E2535", "tickcolor": "#4A5468"},
                    "colorway": ["#00D4AA", "#4B9FFF", "#FFB547", "#FF4B6E", "#A78BFA", "#34D399"]
                }
            }

            try:
                clean_chart_df = df[~df[x_axis].isin(['N/A'])]
                clean_chart_df = clean_chart_df[~clean_chart_df[y_axis].isin(['N/A'])]

                if chart_type == "Bar":
                    fig = px.bar(clean_chart_df, x=x_axis, y=y_axis, color=x_axis,
                                 title=f"{y_axis} by {x_axis}",
                                 color_discrete_sequence=["#00D4AA","#4B9FFF","#FFB547","#FF4B6E","#A78BFA","#34D399"])
                elif chart_type == "Pie":
                    fig = px.pie(clean_chart_df, names=x_axis,
                                 title=f"Distribution of {x_axis}",
                                 color_discrete_sequence=["#00D4AA","#4B9FFF","#FFB547","#FF4B6E","#A78BFA","#34D399"])
                else:
                    fig = px.scatter(clean_chart_df, x=x_axis, y=y_axis, color=x_axis,
                                     title=f"{y_axis} vs {x_axis}",
                                     color_discrete_sequence=["#00D4AA","#4B9FFF","#FFB547","#FF4B6E","#A78BFA","#34D399"])

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0D1119",
                    font=dict(family="IBM Plex Sans, sans-serif", color="#8B95AE", size=11),
                    title_font=dict(family="Syne, sans-serif", color="#EDF0F7", size=13),
                    xaxis=dict(gridcolor="#1E2535", linecolor="#1E2535"),
                    yaxis=dict(gridcolor="#1E2535", linecolor="#1E2535"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E2535"),
                    margin=dict(t=50, b=30, l=20, r=20)
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning("Visualization failed. Ensure the Y-axis contains continuous numerical data.")
