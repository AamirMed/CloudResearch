import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import base64
import json
import re
import random
import string
import time # Added for Rate Limit Throttling
from groq import Groq
from openai import OpenAI 
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image
import fitz  
import io
import plotly.express as px

# --- 1. UI SETUP & PREMIUM CSS INJECTION ---
st.set_page_config(page_title="CloudResearch Command Center", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# PREMIUM CSS: Transforms the UI from "Raw" to "Enterprise Clinical"
st.markdown("""
    <style>
        /* 1. Global Typography & Canvas */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="st-emotion-cache"] {
            font-family: 'Inter', sans-serif;
            background-color: #F4F7F9 !important;
        }
        
        /* 2. Floating Workspace Container */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 92% !important;
            background-color: #FFFFFF !important;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05) !important;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* 3. Dark Sleek Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1E293B !important;
            border-right: none !important;
        }
        section[data-testid="stSidebar"] * {
            color: #F8FAFC !important;
        }
        /* Style input boxes inside the dark sidebar */
        section[data-testid="stSidebar"] .stTextInput input, section[data-testid="stSidebar"] .stTextArea textarea {
            background-color: #334155 !important;
            color: white !important;
            border: 1px solid #475569 !important;
        }

        /* 4. Typography Polish */
        h1 { 
            font-size: 2rem !important; 
            font-weight: 700 !important; 
            color: #0F172A !important; 
            letter-spacing: -0.8px !important; 
            border-bottom: 2px solid #F1F5F9; 
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }
        h2 { font-size: 1.4rem !important; font-weight: 600 !important; color: #1E293B !important; margin-top: 1.5rem !important; }
        h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #475569 !important; }
        
        /* 5. Modern Buttons */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            height: 3rem !important;
        }
        /* Make 'Primary' buttons pop */
        div[data-testid="stForm"] .stButton > button {
            width: 100% !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(46, 102, 246, 0.2) !important;
        }
        
        /* 6. Beautiful File Uploader */
        [data-testid="stFileUploadDropzone"] {
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            background-color: #F8FAFC !important;
            padding: 2.5rem !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #2E66F6 !important;
            background-color: #EFF6FF !important;
        }
        
        /* 7. Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 2px solid #F1F5F9;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600 !important;
            height: 3.5rem;
            color: #64748B !important;
        }
        .stTabs [aria-selected="true"] {
            color: #2E66F6 !important;
            border-bottom: 2px solid #2E66F6 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---
@st.cache_resource
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

def sync_user_profile(username, mode="pull", profile_data=None):
    admin_url = st.secrets.get("ADMIN_SHEET_URL", "")
    if not admin_url: return None
    client = get_google_sheet_client()
    try:
        sheet = client.open_by_url(admin_url).worksheet("Profiles")
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = client.open_by_url(admin_url)
        sheet = spreadsheet.add_worksheet(title="Profiles", rows="100", cols="10")
        headers = ["Username", "Target_Sheet_URL", "Schema", "Prompt", "Abbreviations", "Extra_Rules", "Anti_Rules"]
        sheet.update(range_name="A1:G1", values=[headers])

    if mode == "pull":
        records = sheet.get_all_records()
        for row in records:
            if str(row.get("Username")) == username: return row
        return None 
    elif mode == "push" and profile_data:
        records = sheet.get_all_records()
        row_idx = None
        for i, row in enumerate(records):
            if str(row.get("Username")) == username:
                row_idx = i + 2 
                break
        row_values = [username, profile_data.get("Target_Sheet_URL", ""), profile_data.get("Schema", ""), profile_data.get("Prompt", ""), profile_data.get("Abbreviations", ""), profile_data.get("Extra_Rules", ""), profile_data.get("Anti_Rules", "")]
        if row_idx: sheet.update(range_name=f"A{row_idx}:G{row_idx}", values=[row_values])
        else: sheet.append_row(row_values)
        return True

def sync_with_google_sheets(local_dataframe, sheet_url, tab_name, mode="pull"):
    google_client = get_google_sheet_client()
    try:
        sheet = google_client.open_by_url(sheet_url).worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = google_client.open_by_url(sheet_url)
        sheet = spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="20")
    
    if mode == "pull":
        cloud_data = sheet.get_all_values()
        if len(cloud_data) > 1: cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
        elif len(cloud_data) == 1: cloud_df = pd.DataFrame(columns=cloud_data[0])
        else: cloud_df = pd.DataFrame()

        if not cloud_df.empty:
            cloud_df.rename(columns=lambda x: 'System_ID' if str(x).strip().lower() in ['system_id', 'system id'] else x, inplace=True)
            if 'System_ID' not in cloud_df.columns:
                existing = set()
                new_ids = [generate_unique_id(existing) for _ in range(len(cloud_df))]
                cloud_df.insert(0, 'System_ID', new_ids)
            else:
                missing_mask = cloud_df['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
                if missing_mask.any():
                    existing_valid = set(cloud_df.loc[~missing_mask, 'System_ID'].astype(str).tolist())
                    cloud_df.loc[missing_mask, 'System_ID'] = [generate_unique_id(existing_valid) for _ in range(missing_mask.sum())]
        return cloud_df
    elif mode == "push":
        sheet.clear()
        if not local_dataframe.empty:
            df_to_upload = local_dataframe.copy().astype(str)
            cols = ['System_ID'] + [c for c in df_to_upload.columns if c != 'System_ID']
            df_to_upload = df_to_upload[cols]
            data_to_upload = [df_to_upload.columns.values.tolist()] + df_to_upload.values.tolist()
            sheet.update(range_name="A1", values=data_to_upload)
        return local_dataframe

def compress_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L') # Grayscale
    img.thumbnail((768, 768)) # Resize
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=75)
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
    # MINIFIED TOKEN-SAVING PROMPT
    return f"Task: Extract clinical data to JSON array. Prompt: {user_prompt}. Map: {abbreviations}. Rules: {extra_rules}. Avoid: {anti_rules}. Output RAW JSON array `[{{...}}]` ONLY. Use 'N/A' if missing."

def blueprint_decoder(image_bytes, columns, final_prompt, model_choice):
    full_prompt = f"{final_prompt}\n\nREQUIRED COLUMNS (JSON KEYS): [{columns}]\n\nOutput a valid JSON ARRAY format."
    compressed_bytes = compress_image(image_bytes)
    base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
    raw_output = ""
    try:
        if "Gemini" in model_choice:
            img = Image.open(io.BytesIO(compressed_bytes))
            model = genai.GenerativeModel(st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash"))
            response = model.generate_content([full_prompt, img], generation_config={"response_mime_type": "application/json"})
            raw_output = response.text.strip()
        elif "Groq" in model_choice:
            client = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview", 
                messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            raw_output = response.choices[0].message.content.strip()
        elif "OpenAI" in model_choice:
            client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt + " Wrap array in key 'data'."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            parsed = json.loads(response.choices[0].message.content)
            raw_output = json.dumps(parsed.get("data", []))

        # JSON Cleaning
        if "
http://googleusercontent.com/immersive_entry_chip/0

Would you like me to show you how to add a **"Status Dashboard"** at the top of the app that counts total patients and missing data fields in real-time?
