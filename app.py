import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import base64
import json
import re
from groq import Groq
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image
import io

# --- 1. UI SETUP ---
st.set_page_config(page_title="CloudResearch Command Center", layout="wide", page_icon="☁️")

# --- 2. HELPER FUNCTIONS ---
def get_google_sheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

def sync_with_google_sheets(local_dataframe, sheet_url, tab_name):
    google_client = get_google_sheet_client()
    
    try:
        sheet = google_client.open_by_url(sheet_url).worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = google_client.open_by_url(sheet_url)
        sheet = spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="20")
    
    # 1. PULL CLOUD DATA
    cloud_data = sheet.get_all_values()
    if len(cloud_data) > 1:
        cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
    elif len(cloud_data) == 1:
        cloud_df = pd.DataFrame(columns=cloud_data[0])
    else:
        cloud_df = pd.DataFrame()

    # 2. GRANDFATHER IN OLD DATA (Assign IDs to existing Google Sheet rows)
    if not cloud_df.empty:
        # Standardize the ID column name
        cloud_df.rename(columns=lambda x: 'System_ID' if str(x).strip().lower() in ['system_id', 'system id'] else x, inplace=True)
        
        if 'System_ID' not in cloud_df.columns:
            cloud_df.insert(0, 'System_ID', [f"CR-{1000 + i}" for i in range(len(cloud_df))])
        else:
            # Fill in any blanks for rows that were typed manually into Google Sheets
            missing_mask = cloud_df['System_ID'].astype(str).str.strip().isin(['', 'nan', 'None', 'N/A'])
            if missing_mask.any():
                valid_ids = cloud_df.loc[~missing_mask, 'System_ID'].astype(str).str.extract(r'(\d+)').dropna().astype(int)
                start_val = int(valid_ids.max()[0]) + 1 if not valid_ids.empty else 1000
                
                new_ids = []
                for _ in range(missing_mask.sum()):
                    new_ids.append(f"CR-{start_val}")
                    start_val += 1
                cloud_df.loc[missing_mask, 'System_ID'] = new_ids

    # 3. MERGE CLOUD AND LOCAL
    if not local_dataframe.empty:
        combined_df = pd.concat([cloud_df, local_dataframe], ignore_index=True)
        combined_df['System_ID'] = combined_df['System_ID'].astype(str).str.strip()
        combined_df = combined_df[combined_df['System_ID'] != '']
        combined_df = combined_df[combined_df['System_ID'] != 'nan']
        combined_df.replace({'N/A': np.nan, 'NA': np.nan, '': np.nan, 'None': np.nan}, inplace=True)
        # Keep the most recently edited version if there are duplicates
        combined_df = combined_df.groupby('System_ID', as_index=False).last() 
        combined_df.fillna('N/A', inplace=True)
    else:
        combined_df = cloud_df
        
    # 4. PUSH BACK TO CLOUD
    sheet.clear()
    if not combined_df.empty:
        combined_df = combined_df.astype(str)
        cols = ['System_ID'] + [c for c in combined_df.columns if c != 'System_ID']
        combined_df = combined_df[cols]
        data_to_upload = [combined_df.columns.values.tolist()] + combined_df.values.tolist()
        sheet.update(range_name="A1", values=data_to_upload)
    
    return combined_df

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
    prompt = f"{system_instructions}\nExtract data from this medical document. REQUIRED EXACT KEYS: [{columns}]\nUSER RULES: {rules}\nSTRICT JSON PROTOCOL: Output EXACTLY ONE valid JSON ARRAY format: [{{...}}, {{...}}]. The keys MUST exactly match the REQUIRED KEYS. If a value is missing, output 'N/A'."
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
    )
    
    raw_output = response.choices[0].message.content.strip()
    match = re.search(r'\[.*\]', raw_output, re.DOTALL)
    return match.group(0) if match else raw_output

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
    # 🎉 THEY ARE LOGGED IN! THE APP UNLOCKS.
    username = st.session_state.get("username")
    name = st.session_state.get("name")
    
    user_prefs = st.secrets["credentials"]["usernames"][username]
    saved_sheet_url = user_prefs.get("sheet_url", "")
    saved_schema = user_prefs.get("default_schema", "Age, Gender, Organism")

    # --- AUTO-SYNC ON LOGIN ---
    if "master_database" not in st.session_state:
        st.session_state.master_database = pd.DataFrame()
        if saved_sheet_url and username:
            try:
                # Silently fetch the Google Sheet into memory immediately
                st.session_state.master_database = sync_with_google_sheets(pd.DataFrame(), saved_sheet_url, username)
            except Exception:
                pass # If it fails on first load, just proceed with empty memory

    with st.sidebar:
        st.success(f"Welcome back, {name}!")
        authenticator.logout("Logout", "sidebar")
        st.divider()
        
        st.header("📋 1. Your Schema")
        st.warning("Do NOT type 'System_ID' here.")
        column_headers = st.text_input("Exact Clinical Columns:", saved_schema)
        extra_rules = st.text_area("Specific Rules:", "If marked S write Sensitive. If R write Resistant. Strip colons.")
        
        st.divider()
        st.header("🔗 2. Database Connection")
        user_sheet_url = st.text_input("Google Sheet URL:", saved_sheet_url)
        project_tab = username 
        st.caption(f"Routing data to secure tab: **{project_tab}**")
        
        st.divider()
        st.header("💾 3. Local Controls")
        if st.button("🚨 Clear Local Memory"):
            st.session_state.master_database = pd.DataFrame()
            st.rerun()

    st.title("☁️ CloudResearch Command Center")

    # --- 5. DATA ENTRY MODE ---
    st.subheader("📸 Step 1: Data Entry Mode")
    entry_mode = st.radio("Select your workflow:", ["🆕 Add New Patients", "🔄 Update Existing Patient"], index=0)
    st.divider()
    expected_cols = [c.strip() for c in column_headers.split(',')]

    if "Add New" in entry_mode:
        st.info("Upload rosters or reports. The app will generate unique IDs automatically.")
        with st.form("add_new_form", clear_on_submit=True):
            uploaded_files = st.file_uploader("Upload patient documents:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
            submitted = st.form_submit_button("⚙️ Extract & Generate IDs", type="primary")

        if submitted and uploaded_files:
            patient_dfs = []
            for file in uploaded_files:
                with st.spinner(f"AI is reading {file.name}..."):
                    try:
                        raw_json = blueprint_decoder(file.getvalue(), column_headers, extra_rules, st.secrets["GROQ_API_KEY"])
                        ai_data_list = json.loads(raw_json) 
                        if isinstance(ai_data_list, dict):
                            ai_data_list = [ai_data_list] 
                        for patient_data in ai_data_list:
                            filtered_data = {col: patient_data.get(col, patient_data.get(f"{col}:", 'N/A')) for col in expected_cols}
                            patient_dfs.append(pd.DataFrame([filtered_data]))
                    except Exception as e:
                        st.error(f"Could not read {file.name}. Error: {e}")
            
            if patient_dfs:
                current_batch_df = pd.concat(patient_dfs, ignore_index=True)
                
                # --- SMARTER ID GENERATION ---
                if st.session_state.master_database.empty or 'System_ID' not in st.session_state.master_database.columns:
                    start_num = 1000
                else:
                    existing_ids = st.session_state.master_database['System_ID'].astype(str)
                    nums = existing_ids.str.extract(r'(\d+)').dropna().astype(int)
                    start_num = int(nums.max()[0]) + 1 if not nums.empty else 1000
                
                new_ids = [f"CR-{start_num + i}" for i in range(len(current_batch_df))]
                current_batch_df.insert(0, "System_ID", new_ids)
                
                if not st.session_state.master_database.empty:
                    st.session_state.master_database = pd.concat([st.session_state.master_database, current_batch_df], ignore_index=True)
                else:
                    st.session_state.master_database = current_batch_df
                st.success(f"✅ Extracted {len(current_batch_df)} new patients! IDs assigned: {new_ids[0]} to {new_ids[-1]}")

    elif "Update" in entry_mode:
        st.info("Check your Google Sheet for the patient's 'System_ID'. Type it below to add new data.")
        with st.form("update_form", clear_on_submit=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                target_id = st.text_input("Exact System_ID to update:", placeholder="CR-1005")
            with col2:
                update_files = st.file_uploader("Upload new documents:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
            update_submitted = st.form_submit_button("🔄 Update Patient Profile", type="primary")

        if update_submitted:
            if not target_id:
                st.error("You must provide the System_ID.")
            elif st.session_state.master_database.empty or target_id not in st.session_state.master_database['System_ID'].values:
                st.error(f"❌ Could not find '{target_id}' in local memory. Did you Sync from the cloud first?")
            elif not update_files:
                st.error("Please upload the documents.")
            else:
                for file in update_files:
                    with st.spinner(f"Reading new data for {target_id}..."):
                        try:
                            raw_json = blueprint_decoder(file.getvalue(), column_headers, extra_rules, st.secrets["GROQ_API_KEY"])
                            ai_data = json.loads(raw_json)
                            if isinstance(ai_data, list):
                                ai_data = ai_data[0] 
                            
                            idx = st.session_state.master_database.index[st.session_state.master_database['System_ID'] == target_id].tolist()[0]
                            for col in expected_cols:
                                new_val = ai_data.get(col, 'N/A')
                                if new_val != 'N/A': 
                                    st.session_state.master_database.at[idx, col] = new_val
                            st.success(f"✅ Profile {target_id} successfully updated!")
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")

    # --- 6. HUMAN-IN-THE-LOOP EDITOR ---
    if not st.session_state.master_database.empty:
        st.divider()
        st.subheader("📝 Step 2: Verify & Edit Data")
        st.session_state.master_database = st.data_editor(
            st.session_state.master_database, 
            num_rows="dynamic", 
            use_container_width=True,
            key="data_verifier"
        )

    # --- 7. CLOUD SYNC ---
    st.divider()
    st.subheader("🌐 Step 3: Finalize & Sync")
    col_x, col_y = st.columns([1, 1])

    with col_x:
        if st.button("🚀 PUSH / PULL GOOGLE CLOUD", type="primary", use_container_width=True):
            with st.spinner("Syncing secure data..."):
                if not user_sheet_url or not project_tab:
                    st.error("❌ Please ensure your Database Connection in the sidebar is filled out.")
                else:
                    try:
                        merged_data = sync_with_google_sheets(st.session_state.master_database, user_sheet_url, project_tab)
                        st.session_state.master_database = merged_data
                        st.toast("✅ Sync Complete! Data secured.", icon="☁️")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Sync Failed. Error: {e}")

    with col_y:
        if not st.session_state.master_database.empty:
            csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Offline CSV Backup", data=csv_data, file_name=f"{project_tab}_backup.csv", mime="text/csv", use_container_width=True)
