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
from openai import OpenAI # Universal remote for OpenAI, OpenRouter (Llama & Qwen)
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image
import fitz  
import io
import plotly.express as px

# --- 1. UI SETUP & GLOBAL CONFIG ---
st.set_page_config(page_title="CloudResearch Command Center", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.markdown("""
    <style>
        h1 { font-size: 1.5rem !important; font-weight: 700 !important; padding-bottom: 0.5rem !important; }
        h2 { font-size: 1.1rem !important; font-weight: 600 !important; padding-top: 1rem !important; padding-bottom: 0.2rem !important; }
        h3 { font-size: 1.05rem !important; font-weight: 600 !important; padding-bottom: 0.2rem !important; }
        .streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.95rem !important; }
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

# --- MASTER ADMIN DIRECTORY ENGINE ---
def sync_user_profile(username, mode="pull", profile_data=None):
    admin_url = st.secrets.get("ADMIN_SHEET_URL", "")
    if not admin_url:
        return None

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
            if str(row.get("Username")) == username:
                return row
        return None 
        
    elif mode == "push" and profile_data:
        records = sheet.get_all_records()
        row_idx = None
        for i, row in enumerate(records):
            if str(row.get("Username")) == username:
                row_idx = i + 2 
                break
                
        row_values = [
            username,
            profile_data.get("Target_Sheet_URL", ""),
            profile_data.get("Schema", ""),
            profile_data.get("Prompt", ""),
            profile_data.get("Abbreviations", ""),
            profile_data.get("Extra_Rules", ""),
            profile_data.get("Anti_Rules", "")
        ]
        
        if row_idx:
            sheet.update(range_name=f"A{row_idx}:G{row_idx}", values=[row_values])
        else:
            sheet.append_row(row_values)
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
        return local_dataframe

def compress_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    
    # TOKEN DIET: Grayscale Conversion + Resolution Reduction
    img = img.convert('L') 
    img.thumbnail((1024, 1024)) 
    
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

# --- THE NEW 5-MODEL BLUEPRINT DECODER ---
def blueprint_decoder(image_bytes, columns, final_prompt, model_choice):
    full_prompt = f"{final_prompt}\n\nREQUIRED COLUMNS (JSON KEYS): [{columns}]\n\nOutput a valid JSON ARRAY format: [{{...}}, {{...}}]. If the image contains multiple patients, create a separate JSON object for EACH patient. If you cannot read the image, output an empty array []."
    
    compressed_bytes = compress_image(image_bytes)
    base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
    
    raw_output = ""
    
    try:
        if "Gemini" in model_choice:
            img = Image.open(io.BytesIO(compressed_bytes))
            gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
            model = genai.GenerativeModel(gemini_model_name)
            response = model.generate_content([full_prompt, img], generation_config={"response_mime_type": "application/json"})
            raw_output = response.text.strip()
            
        elif "Groq" in model_choice:
            client = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
            groq_prompt = full_prompt + " ONLY output the raw JSON array."
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview", 
                messages=[{"role": "user", "content": [{"type": "text", "text": groq_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            raw_output = response.choices[0].message.content.strip()

        elif "OpenAI" in model_choice:
            client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt + " Wrap the array in a JSON object with the key 'data'."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            parsed = json.loads(response.choices[0].message.content)
            raw_output = json.dumps(parsed.get("data", []))

        elif "Llama" in model_choice and "OpenRouter" in model_choice:
            client = OpenAI(
                api_key=st.secrets.get("OPENROUTER_API_KEY", ""), 
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model="meta-llama/llama-3.2-90b-vision-instruct:free",
                messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            raw_output = response.choices[0].message.content.strip()

        elif "Qwen" in model_choice and "OpenRouter" in model_choice:
            client = OpenAI(
                api_key=st.secrets.get("OPENROUTER_API_KEY", ""), 
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model="qwen/qwen-2-vl-72b-instruct:free",
                messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            raw_output = response.choices[0].message.content.strip()

        # Universal JSON cleaner 
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1]
        if "```" in raw_output:
            raw_output = raw_output.split("```")[0]
            
        match = re.search(r'(\[.*\]|\{.*\})', raw_output.strip(), re.DOTALL)
        return match.group(1) if match else raw_output.strip()

    except Exception as e:
        st.error(f"{model_choice} Connection Error: Please verify your API keys in the secrets file. Details: {e}")
        return "[]"


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
    st.error(f"Authentication System Error: The library failed to load.")
    st.error(f"Developer details: {e}")
    st.stop()

# --- 4. SECURE PROFILE LOADING & LOGIC ---
auth_status = st.session_state.get("authentication_status")

if auth_status == False:
    st.error("Username or password is incorrect.")
elif auth_status == None:
    st.title("CloudResearch")
    st.warning("Please enter your credentials to access the Command Center.")

elif auth_status == True:
    username = st.session_state.get("username")
    name = st.session_state.get("name")
    
    # 1. PULL MASTER PROFILE FROM CLOUD ONCE PER SESSION
    if "profile_loaded" not in st.session_state:
        with st.spinner("Connecting to secure profile..."):
            cloud_profile = sync_user_profile(username, mode="pull")
            
            if cloud_profile:
                st.session_state.target_sheet_url = cloud_profile.get("Target_Sheet_URL", "")
                st.session_state.schema_input = cloud_profile.get("Schema", "Age, Gender, Organism")
                st.session_state.user_prompt = cloud_profile.get("Prompt", "You are an expert clinical researcher. Extract structured medical data from the provided documents.")
                st.session_state.abbreviations = cloud_profile.get("Abbreviations", "")
                st.session_state.extra_rules = cloud_profile.get("Extra_Rules", "")
                st.session_state.anti_rules = cloud_profile.get("Anti_Rules", "")
            else:
                st.session_state.target_sheet_url = ""
                st.session_state.schema_input = "Age, Gender, Organism"
                st.session_state.user_prompt = "You are an expert clinical researcher. Extract structured medical data from the provided documents."
                st.session_state.abbreviations = ""
                st.session_state.extra_rules = ""
                st.session_state.anti_rules = ""
                
            st.session_state.profile_loaded = True

    # 2. PULL PATIENT DATABASE
    if "master_database" not in st.session_state:
        st.session_state.master_database = pd.DataFrame()
        if st.session_state.target_sheet_url:
            try:
                pulled_df = sync_with_google_sheets(pd.DataFrame(), st.session_state.target_sheet_url, username, mode="pull")
                st.session_state.master_database = pulled_df
            except Exception as e:
                st.error(f"Background sync failed to connect to Google Sheets: {e}") 

    with st.sidebar:
        st.success(f"Session Active: {name}")
        authenticator.logout("Logout", "sidebar")
        st.divider()
        
        st.header("1. Processing Engine")
        
        # --- THE UPDATED 5-MODEL SELECTOR ---
        selected_model = st.selectbox(
            "Model Selection:", 
            [
                "Google Gemini (Flash)", 
                "Groq (Llama Vision)", 
                "OpenRouter (Free Meta Llama)", 
                "OpenRouter (Free Qwen Vision)", 
                "OpenAI (GPT-4o-Mini)"
            ]
        )
        st.divider()
        
        st.header("2. Database Connection")
        active_sheet_url = st.text_input("Google Sheet URL:", value=st.session_state.target_sheet_url)
        st.session_state.target_sheet_url = active_sheet_url
        project_tab = username 
        
        st.divider()
        
        st.header("3. Target Schema")
        st.info("System IDs are auto-generated to prevent duplicate records and ensure safe cloud synchronization.")
        
        if "safe_schema_val" not in st.session_state:
            st.session_state.safe_schema_val = st.session_state.schema_input

        col_pull, col_push = st.columns(2)
        with col_pull:
            if st.button("Pull Schema", use_container_width=True):
                if active_sheet_url:
                    with st.spinner("Synchronizing..."):
                        try:
                            sheet = get_google_sheet_client().open_by_url(active_sheet_url).worksheet(project_tab)
                            cloud_headers = sheet.row_values(1)
                            if cloud_headers:
                                clean_headers = [c for c in cloud_headers if c.lower() != 'system_id']
                                st.session_state.safe_schema_val = ", ".join(clean_headers)
                                st.toast("Schema pulled successfully.")
                                st.rerun()
                        except Exception as e:
                            st.toast(f"Sync error: {e}")

        with col_push:
            if st.button("Push Schema", use_container_width=True):
                if active_sheet_url:
                    with st.spinner("Synchronizing..."):
                        try:
                            sheet = get_google_sheet_client().open_by_url(active_sheet_url).worksheet(project_tab)
                            current_cols = [c.strip() for c in st.session_state.safe_schema_val.split(',') if c.strip()]
                            final_headers = ['System_ID'] + [c for c in current_cols if c.lower() != 'system_id']
                            sheet.update(range_name="A1", values=[final_headers]) 
                            st.toast("Schema pushed successfully.")
                        except Exception as e:
                            st.toast(f"Sync error: {e}")
        
        updated_schema = st.text_input("Define Schema Columns:", value=st.session_state.safe_schema_val)
        st.session_state.safe_schema_val = updated_schema
        st.session_state.schema_input = updated_schema
        
        st.divider()
        
        st.header("4. Extraction Logic")
        st.session_state.user_prompt = st.text_area(
            "Primary Directive", 
            value=st.session_state.user_prompt
        )
        
        with st.expander("Advanced Extraction Constraints"):
            st.session_state.abbreviations = st.text_area("Abbreviations Map", value=st.session_state.abbreviations)
            st.session_state.extra_rules = st.text_area("Inclusion Rules", value=st.session_state.extra_rules)
            st.session_state.anti_rules = st.text_area("Exclusion Rules", value=st.session_state.anti_rules)
        
        # --- THE MASTER SAVE BUTTON ---
        if st.button("💾 Save Profile Configuration", type="primary", use_container_width=True):
            with st.spinner("Updating Central Directory..."):
                profile_dict = {
                    "Target_Sheet_URL": st.session_state.target_sheet_url,
                    "Schema": st.session_state.schema_input,
                    "Prompt": st.session_state.user_prompt,
                    "Abbreviations": st.session_state.abbreviations,
                    "Extra_Rules": st.session_state.extra_rules,
                    "Anti_Rules": st.session_state.anti_rules
                }
                sync_user_profile(username, mode="push", profile_data=profile_dict)
                st.success("Profile saved permanently!")
        
        st.divider()
        st.header("5. System Controls")
        with st.expander("System Reset"):
            st.warning("Warning: This action clears all unsaved local data.")
            if st.button("Purge Local Cache", use_container_width=True):
                st.session_state.master_database = pd.DataFrame()
                st.rerun()

    st.title("CloudResearch Command Center")

    tabs = st.tabs(["Data Entry & Synchronization", "Clinical Data Explorer"])
    expected_cols = [c.strip() for c in st.session_state.schema_input.split(',') if c.strip()]

    # ==========================================
    # TAB 1: DATA ENTRY & SYNC
    # ==========================================
    with tabs[0]:
        st.subheader("Record Management")
        entry_mode = st.radio(
            "Select Processing Mode:", 
            ["Single Record (Compile Pages)", "Batch Processing (Roster Extract)", "Update Existing Record"], 
            index=0, horizontal=True
        )
        st.divider()

        if "Single Record" in entry_mode:
            st.info("Upload documents for a single subject. The engine will compile data into a unified profile.")
            with st.form("add_single_form", clear_on_submit=True):
                uploaded_files = st.file_uploader("Upload Documents:", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)
                submitted = st.form_submit_button(f"Process via {selected_model.split(' ')[0]}", type="primary")

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

                with st.spinner("Extracting parameters and compiling profile..."):
                    master_patient_data = {col: 'N/A' for col in expected_cols}
                    for image_bytes in ready_images:
                        raw_json = blueprint_decoder(image_bytes, st.session_state.schema_input, final_prompt, selected_model)
                        
                        time.sleep(2) # <--- ADDED THROTTLE: Protects against Rate Limit crashes
                        
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
                    st.success(f"Record compiled successfully. Assigned ID: {new_id}")

        elif "Batch Processing" in entry_mode:
            st.info("Upload rosters or multi-subject reports. The engine will isolate entities and assign unique IDs.")
            with st.form("add_multiple_form", clear_on_submit=True):
                uploaded_files = st.file_uploader("Upload Documents:", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)
                submitted = st.form_submit_button(f"Process Batch via {selected_model.split(' ')[0]}", type="primary")

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
                    with st.spinner(f"Analyzing structure on page {i+1}..."):
                        roster_prompt = final_prompt + "\nCRITICAL: Extract EVERY subject as a separate JSON object in the array."
                        raw_json = blueprint_decoder(image_bytes, st.session_state.schema_input, roster_prompt, selected_model)
                        
                        time.sleep(2) # <--- ADDED THROTTLE: Protects against Rate Limit crashes
                        
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
                    st.success(f"Batch processing complete. Extracted {len(current_batch_df)} records.")

        elif "Update Existing" in entry_mode:
            st.info("Append new documentation to an existing System_ID.")
            with st.form("update_form", clear_on_submit=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_id = st.text_input("System_ID Reference:", placeholder="CR-XXXX")
                with col2:
                    update_files = st.file_uploader("Upload Appendices:", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)
                update_submitted = st.form_submit_button(f"Update Record via {selected_model.split(' ')[0]}", type="primary")

            if update_submitted:
                if not target_id:
                    st.error("System_ID reference is required.")
                elif st.session_state.master_database.empty or target_id not in st.session_state.master_database['System_ID'].values:
                    st.error(f"System_ID '{target_id}' not found in local cache. Synchronize with cloud first.")
                elif not update_files:
                    st.error("Documentation upload required.")
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
                            
                            time.sleep(2) # <--- ADDED THROTTLE: Protects against Rate Limit crashes
                            
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
                            st.success(f"Record {target_id} updated successfully.")

        if not st.session_state.master_database.empty:
            st.divider()
            st.subheader("Data Verification Table")
            st.caption("Manual correction interface prior to cloud commit.")
            st.session_state.master_database = st.data_editor(
                st.session_state.master_database, 
                num_rows="dynamic", 
                use_container_width=True,
                key="data_verifier"
            )

        st.divider()
        st.subheader("Cloud Synchronization")
        col_x, col_y, col_z = st.columns([1, 1, 1])

        with col_x:
            if st.button("Commit to Cloud", type="primary", use_container_width=True):
                if st.session_state.master_database.empty:
                    st.warning("Local cache is empty. Commit aborted.")
                else:
                    missing_ids = st.session_state.master_database['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
                    if missing_ids.any():
                        st.error(f"Validation Error: {missing_ids.sum()} record(s) lack a System_ID.")
                    elif not active_sheet_url or not project_tab:
                        st.error("Database connection parameters missing.")
                    else:
                        with st.spinner("Committing to Google servers..."):
                            try:
                                final_cols = ['System_ID'] + [c for c in expected_cols if c.lower() != 'system_id']
                                for col in final_cols:
                                    if col not in st.session_state.master_database.columns:
                                        st.session_state.master_database[col] = 'N/A'
                                st.session_state.master_database = st.session_state.master_database[final_cols]
                                sync_with_google_sheets(st.session_state.master_database, active_sheet_url, project_tab, mode="push")
                                st.toast("Cloud commit successful.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Commit failed: {e}")
        with col_y:
            if st.button("Pull from Cloud", use_container_width=True):
                with st.spinner("Downloading database instance..."):
                    if not active_sheet_url or not project_tab:
                        st.error("Database connection parameters missing.")
                    else:
                        try:
                            pulled_df = sync_with_google_sheets(pd.DataFrame(), active_sheet_url, project_tab, mode="pull")
                            st.session_state.master_database = pulled_df
                            st.toast("Local cache synchronized with cloud.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Sync failed: {e}")
        with col_z:
            if not st.session_state.master_database.empty:
                csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
                st.download_button("Export Local CSV", data=csv_data, file_name=f"{project_tab}_export.csv", mime="text/csv", use_container_width=True)

    # ==========================================
    # TAB 2: DYNAMIC DATA EXPLORER
    # ==========================================
    with tabs[1]:
        st.subheader("Data Explorer")
        
        if st.session_state.master_database.empty:
            st.info("Synchronize with cloud or process records to enable analytics.")
        else:
            df = st.session_state.master_database.copy()
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip().str.upper()
                    df[col] = df[col].replace({'N/A': 'N/A', 'NAN': 'N/A', 'NONE': 'N/A', '': 'N/A'})
            
            all_columns = df.columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Categorical Axis (X)", all_columns)
            with col2:
                y_axis = st.selectbox("Numerical Axis (Y)", all_columns)

            chart_type = st.radio("Visualization Form", ["Bar", "Pie", "Scatter"], horizontal=True)

            try:
                clean_chart_df = df[~df[x_axis].isin(['N/A'])]
                clean_chart_df = clean_chart_df[~clean_chart_df[y_axis].isin(['N/A'])]

                if chart_type == "Bar":
                    fig = px.bar(clean_chart_df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Pie":
                    fig = px.pie(clean_chart_df, names=x_axis, title=f"Distribution of {x_axis}")
                else:
                    fig = px.scatter(clean_chart_df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} vs {x_axis}")

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Visualization failed. Ensure the selected Y-axis contains continuous numerical data.")
