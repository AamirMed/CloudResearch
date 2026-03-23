# -*- coding: utf-8 -*-
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
st.set_page_config(page_title="CloudResearch", layout="wide", page_icon=None)

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ============================================================
# CLOUDRESEARCH -- PRECISION DARK UI SYSTEM
# Typography: IBM Plex Sans (body) + Syne (display)
# Palette: Deep navy base, electric teal accent, slate surfaces
# ============================================================
with open("style.css", "r") as _f:
    st.markdown(_f.read(), unsafe_allow_html=True)


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

    # -- SIDEBAR ---------------------------------------------
    with st.sidebar:

        # Logo mark
        st.markdown(f"""
            <div class="sidebar-logo">
                <div class="sidebar-hexmark">#</div>
                <div>
                    <div class="sidebar-wordmark">Cloud<span>Research</span></div>
                    <div class="sidebar-version">v2.0 - CLINICAL</div>
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
                <span style="color:#2A3550;">#</span> 
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
            if st.button("Pull Schema", use_container_width=True):
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
            if st.button("Push Schema", use_container_width=True):
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
                placeholder="DM -> Diabetes Mellitus\nHTN -> Hypertension\nS -> Sensitive\nR -> Resistant",
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
        with st.expander("Danger Zone"):
            st.warning("Clears all unsaved local data permanently.")
            if st.button("Purge Local Cache", type="primary", use_container_width=True):
                st.session_state.master_database = pd.DataFrame()
                st.rerun()


    # -- MAIN HEADER -----------------------------------------
    record_count = len(st.session_state.master_database) if not st.session_state.master_database.empty else 0
    model_short = selected_model.split(" ")[0]

    st.markdown(f"""
        <div style="display:flex; align-items:flex-end; justify-content:space-between; 
                    padding-bottom:1.2rem; border-bottom:1px solid #1E2535; margin-bottom:1.5rem;">
            <div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#4A5468; 
                            letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.4rem;">
                    # CLOUDRESEARCH - COMMAND CENTER
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
            st.info("Upload all pages for a single subject. 
