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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image
import io
import plotly.express as px

# --- 1. UI SETUP ---
st.set_page_config(page_title="CloudResearch Command Center", layout="wide", page_icon="☁️")

# --- 2. HELPER FUNCTIONS ---
def get_google_sheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

# SECURE 4-DIGIT ID GENERATOR
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
        sheet.clear()
        if not local_dataframe.empty:
            df_to_upload = local_dataframe.copy()
            df_to_upload = df_to_upload.astype(str)
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
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((2048, 2048)) 
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()

def blueprint_decoder(image_bytes, columns, rules, api_key):
    client = Groq(api_key=api_key)
    compressed_bytes = compress_image(image_bytes)
    base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
    
    system_instructions = "You are an expert clinical data extractor. Map informal abbreviations to standard nomenclature."
    prompt = f"{system_instructions}\nExtract data from this medical document. REQUIRED EXACT KEYS: [{columns}]\nUSER RULES: {rules}\nSTRICT JSON PROTOCOL: Output a valid JSON ARRAY format: [{{...}}, {{...}}]. If the image contains multiple patients, create a separate JSON object for EACH patient. NO markdown, NO conversational text. ONLY output the raw JSON array. If you cannot read the image, output an empty array []."
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
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
    st.error(f"⚠️ Authentication System Error: The library failed to load.")
    st.error(f"Developer details: {e}")
    st.stop()

# --- 4. LOGIN LOGIC ---
auth_status = st.session_state.get("authentication_status")

if auth_status == False:
    st.error("❌ Username or password is incorrect.")
elif auth_status == None:
    st.title("☁️ CloudResearch")
    st.warning("🔒 Please enter your username and password to access the Command Center.")

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
            except Exception:
                pass 

    with st.sidebar:
        st.success(f"Welcome back, {name}!")
        authenticator.logout("Logout", "sidebar")
        st.divider()
        
        # --- 1. Database Connection ---
        st.header("🔗 1. Database Connection")
        user_sheet_url = st.text_input("Google Sheet URL:", saved_sheet_url)
        project_tab = username 
        st.caption(f"Routing data to secure tab: **{project_tab}**")
        
        st.divider()
        
        # --- 2. SCHEMA DEFINITION & SYNC ---
        st.header("📋 2. Your Schema")
        st.warning("Do NOT type 'System_ID' here.")
        
        # FIX: Decoupled widget to allow programmatic updates
        updated_schema = st.text_input("Exact Clinical Columns:", value=st.session_state.schema_input)
        st.session_state.schema_input = updated_schema
        
        # Dedicated Schema Sync Buttons
        col_pull, col_push = st.columns(2)
        with col_pull:
            if st.button("⬇️ Pull Cols", use_container_width=True, help="Fetch existing headers from your sheet"):
                if user_sheet_url and project_tab:
                    try:
                        sheet = get_google_sheet_client().open_by_url(user_sheet_url).worksheet(project_tab)
                        cloud_headers = sheet.row_values(1)
                        if cloud_headers:
                            clean_headers = [c for c in cloud_headers if c.lower() != 'system_id']
                            st.session_state.schema_input = ", ".join(clean_headers)
                            st.rerun()
                        else:
                            st.toast("Sheet is completely empty.", icon="⚠️")
                    except Exception as e:
                        st.toast(f"Error pulling columns: {e}", icon="❌")
                else:
                    st.toast("Enter Sheet URL above first.", icon="⚠️")

        with col_push:
            if st.button("⬆️ Push Cols", use_container_width=True, help="Overwrite row 1 of your sheet with these columns"):
                if user_sheet_url and project_tab:
                    try:
                        sheet = get_google_sheet_client().open_by_url(user_sheet_url).worksheet(project_tab)
                        current_cols = [c.strip() for c in st.session_state.schema_input.split(',') if c.strip()]
                        final_headers = ['System_ID'] + [c for c in current_cols if c.lower() != 'system_id']
                        sheet.update(range_name="A1", values=[final_headers]) 
                        st.toast("✅ Cloud headers updated!", icon="☁️")
                    except Exception as e:
                        st.toast(f"Error pushing columns: {e}", icon="❌")
                else:
                    st.toast("Enter Sheet URL above first.", icon="⚠️")
                    
        extra_rules = st.text_area("Specific Rules:", "If marked S write Sensitive. If R write Resistant. Strip colons.")
        
        st.divider()
        st.header("💾 3. Local Controls")
        if st.button("🚨 Clear Local Memory"):
            st.session_state.master_database = pd.DataFrame()
            st.rerun()

    st.title("☁️ CloudResearch Command Center")

    # --- TABBED INTERFACE ---
    tabs = st.tabs(["📸 Data Entry, Verification & Sync", "📊 Data Explorer"])
    expected_cols = [c.strip() for c in st.session_state.schema_input.split(',') if c.strip()]

    # ==========================================
    # TAB 1: DATA ENTRY, VERIFICATION & SYNC
    # ==========================================
    with tabs[0]:
        st.subheader("Add or Update Patients")
        entry_mode = st.radio(
            "Select your workflow:", 
            ["🆕 Single Patient (Merge Pages)", "📋 Multiple Patients (Roster)", "🔄 Update Existing"], 
            index=0, horizontal=True
        )
        st.divider()

        if "Single Patient" in entry_mode:
            st.info("Upload all pages for a SINGLE patient. The AI will merge the data and assign ONE ID.")
            with st.form("add_single_form", clear_on_submit=True):
                uploaded_files = st.file_uploader("Upload patient documents:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
                submitted = st.form_submit_button("⚙️ Extract & Generate 1 ID", type="primary")

            if submitted and uploaded_files:
                with st.spinner(f"AI is reading {len(uploaded_files)} pages and compiling the profile..."):
                    master_patient_data = {col: 'N/A' for col in expected_cols}
                    for file in uploaded_files:
                        raw_json = blueprint_decoder(file.getvalue(), st.session_state.schema_input, extra_rules, st.secrets["GROQ_API_KEY"])
                        try:
                            ai_data_list = json.loads(raw_json) 
                        except json.JSONDecodeError:
                            st.error(f"❌ AI failed to read **{file.name}**. It did not return valid data.")
                            with st.expander("See what the AI actually said"):
                                st.text(raw_json)
                            continue

                        if isinstance(ai_data_list, list) and len(ai_data_list) > 0:
                            ai_data = ai_data_list[0] 
                        elif isinstance(ai_data_list, dict):
                            ai_data = ai_data_list
                        else:
                            ai_data = {}

                        for col in expected_cols:
                            new_val = str(ai_data.get(col, ai_data.get(f"{col}:", 'N/A'))).strip()
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
                    st.success(f"✅ Successfully merged {len(uploaded_files)} pages into a single patient! ID assigned: {new_id}")

        elif "Multiple Patients" in entry_mode:
            st.info("Upload rosters or multi-patient reports. The AI will extract EVERY patient and assign unique IDs.")
            with st.form("add_multiple_form", clear_on_submit=True):
                uploaded_files = st.file_uploader("Upload roster documents:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
                submitted = st.form_submit_button("⚙️ Extract Roster & Generate IDs", type="primary")

            if submitted and uploaded_files:
                patient_dfs = []
                for file in uploaded_files:
                    with st.spinner(f"AI is hunting for multiple patients in {file.name}..."):
                        roster_rules = extra_rules + " CRITICAL: Extract EVERY patient as a separate JSON object in the array."
                        raw_json = blueprint_decoder(file.getvalue(), st.session_state.schema_input, roster_rules, st.secrets["GROQ_API_KEY"])
                        try:
                            ai_data_list = json.loads(raw_json)
                        except json.JSONDecodeError:
                            st.error(f"❌ AI failed to read **{file.name}**. It did not return valid data.")
                            with st.expander("See what the AI actually said"):
                                st.text(raw_json)
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
                    st.success(f"✅ Extracted {len(current_batch_df)} patients from the roster! IDs assigned: {new_ids[0]} to {new_ids[-1]}")

        elif "Update" in entry_mode:
            st.info("Check your database for the patient's 'System_ID'. Type it below to add new data.")
            with st.form("update_form", clear_on_submit=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_id = st.text_input("Exact System_ID to update:", placeholder="CR-A4X9")
                with col2:
                    update_files = st.file_uploader("Upload new documents:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
                update_submitted = st.form_submit_button("🔄 Update Patient Profile", type="primary")

            if update_submitted:
                if not target_id:
                    st.error("You must provide the System_ID.")
                elif st.session_state.master_database.empty or target_id not in st.session_state.master_database['System_ID'].values:
                    st.error(f"❌ Could not find '{target_id}' in local memory. Did you Pull from the cloud first?")
                elif not update_files:
                    st.error("Please upload the documents.")
                else:
                    for file in update_files:
                        with st.spinner(f"Reading new data for {target_id}..."):
                            raw_json = blueprint_decoder(file.getvalue(), st.session_state.schema_input, extra_rules, st.secrets["GROQ_API_KEY"])
                            try:
                                ai_data = json.loads(raw_json)
                            except json.JSONDecodeError:
                                st.error(f"❌ AI failed to read **{file.name}**. It did not return valid data.")
                                with st.expander("See what the AI actually said"):
                                    st.text(raw_json)
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
                            st.success(f"✅ Profile {target_id} successfully updated!")

        # --- DATA EDITOR ---
        if not st.session_state.master_database.empty:
            st.divider()
            st.subheader("📝 Verify & Edit Data")
            st.caption("Double-click any cell to manually correct the AI before saving to cloud.")
            st.session_state.master_database = st.data_editor(
                st.session_state.master_database, 
                num_rows="dynamic", 
                use_container_width=True,
                key="data_verifier"
            )

        # --- CLOUD SYNC CONTROLS ---
        st.divider()
        st.subheader("🌐 Cloud Database Management")
        st.info("Align your app's memory with your secure Google Sheet.")
        col_x, col_y, col_z = st.columns([1, 1, 1])

        with col_x:
            if st.button("⬆️ SAVE TO CLOUD", type="primary", use_container_width=True):
                if st.session_state.master_database.empty:
                    st.warning("⚠️ The local database is empty. There is nothing to save.")
                else:
                    missing_ids = st.session_state.master_database['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
                    
                    if missing_ids.any():
                        num_missing = missing_ids.sum()
                        st.error(f"❌ Validation Failed: {num_missing} row(s) are missing a 'System_ID'. Please fix this in the table above before saving.")
                    elif not user_sheet_url or not project_tab:
                        st.error("❌ Please ensure your Database Connection is filled out in the sidebar.")
                    else:
                        with st.spinner("Aligning Columns and Overwriting Cloud..."):
                            try:
                                final_cols = ['System_ID'] + [c for c in expected_cols if c.lower() != 'system_id']
                                
                                for col in final_cols:
                                    if col not in st.session_state.master_database.columns:
                                        st.session_state.master_database[col] = 'N/A'
                                st.session_state.master_database = st.session_state.master_database[final_cols]

                                sync_with_google_sheets(st.session_state.master_database, user_sheet_url, project_tab, mode="push")
                                st.toast("✅ Cloud Updated Successfully!", icon="☁️")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Push Failed. Error: {e}")
        with col_y:
            if st.button("⬇️ PULL FROM CLOUD", use_container_width=True):
                with st.spinner("Reading Google Sheet Columns..."):
                    if not user_sheet_url or not project_tab:
                        st.error("❌ Please ensure your Database Connection is filled out.")
                    else:
                        try:
                            pulled_df = sync_with_google_sheets(pd.DataFrame(), user_sheet_url, project_tab, mode="pull")
                            st.session_state.master_database = pulled_df
                            
                            if not pulled_df.empty:
                                pulled_cols = [c for c in pulled_df.columns if c.lower() != 'system_id']
                                if pulled_cols:
                                    st.session_state.schema_input = ", ".join(pulled_cols)
                                    
                            st.toast("✅ App Columns aligned to Google Sheets!", icon="⬇️")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Pull Failed. Error: {e}")

        with col_z:
            if not st.session_state.master_database.empty:
                csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
                st.download_button("📥 BACKUP TO CSV", data=csv_data, file_name=f"{project_tab}_backup.csv", mime="text/csv", use_container_width=True)

    # ==========================================
    # TAB 2: DYNAMIC DATA EXPLORER
    # ==========================================
    with tabs[1]:
        st.subheader("📊 Dynamic Data Explorer")
        
        if st.session_state.master_database.empty:
            st.info("Upload data or sync from the cloud to view statistics.")
        else:
            df = st.session_state.master_database
            
            all_columns = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis (Categorical)", all_columns)
            with col2:
                y_axis = st.selectbox("Select Y-axis (Numerical)", all_columns)

            chart_type = st.radio("Chart Type", ["Bar", "Pie", "Scatter"], horizontal=True)

            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Pie":
                    fig = px.pie(df, names=x_axis, title=f"Distribution of {x_axis}")
                else:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} vs {x_axis}")

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("⚠️ Could not generate this chart type with the selected columns. Try selecting different axes or ensure your Y-axis contains numerical data.")
