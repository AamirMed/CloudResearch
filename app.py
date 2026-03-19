import streamlit as st
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
import plotly.express as px

# --- 1. UI SETUP ---
st.set_page_config(page_title="CloudResearch Command Center", layout="wide", page_icon="☁️")
st.title("☁️ CloudResearch Command Center")

# --- 2. THE MASTER DATABASE (MEMORY) ---
if "master_database" not in st.session_state:
    st.session_state.master_database = pd.DataFrame()

# --- 3. API SETUPS ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=GROQ_API_KEY)

# 🔄 TWO-WAY SYNC ENGINE (With "Typo Trap" Auto-Create)
def sync_with_google_sheets(local_dataframe, sheet_url, tab_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    google_client = gspread.authorize(creds)
    
    # Gracefully handle missing tabs
    try:
        sheet = google_client.open_by_url(sheet_url).worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = google_client.open_by_url(sheet_url)
        sheet = spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="20")
    
    cloud_data = sheet.get_all_values()
    if len(cloud_data) > 1:
        cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
    elif len(cloud_data) == 1:
        cloud_df = pd.DataFrame(columns=cloud_data[0])
    else:
        cloud_df = pd.DataFrame()
        
    if not cloud_df.empty:
        cloud_df.rename(columns=lambda x: 'System_ID' if str(x).strip().lower() in ['system_id', 'system id'] else x, inplace=True)

    if not local_dataframe.empty:
        combined_df = pd.concat([local_dataframe, cloud_df], ignore_index=True)
        combined_df['System_ID'] = combined_df['System_ID'].astype(str).str.strip()
        combined_df = combined_df[combined_df['System_ID'] != '']
        combined_df = combined_df[combined_df['System_ID'] != 'nan']
        combined_df.replace({'N/A': np.nan, 'NA': np.nan, '': np.nan, 'None': np.nan}, inplace=True)
        # Keep the most recent data for each System_ID
        combined_df = combined_df.groupby('System_ID', as_index=False).last()
        combined_df.fillna('N/A', inplace=True)
    else:
        combined_df = cloud_df
        
    sheet.clear()
    if not combined_df.empty:
        combined_df = combined_df.astype(str)
        # Ensure System_ID is always the first column
        cols = ['System_ID'] + [c for c in combined_df.columns if c != 'System_ID']
        combined_df = combined_df[cols]
        data_to_upload = [combined_df.columns.values.tolist()] + combined_df.values.tolist()
        sheet.update(range_name="A1", values=data_to_upload)
    
    return combined_df

# --- 4. THE BLUEPRINT DECODER (Batch & Normalization Upgrades) ---
def compress_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((2048, 2048)) 
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()

def blueprint_decoder(image_bytes, columns, rules):
    compressed_bytes = compress_image(image_bytes)
    base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
    
    # 🧠 The Invisible Normalizer
    system_instructions = """
    You are an expert clinical data extractor. You must map informal abbreviations to standard medical nomenclature.
    For example, 'Cipro' -> 'Ciprofloxacin', 'Staph A' -> 'Staphylococcus aureus'. 
    """
    
    prompt = f"""
    {system_instructions}
    
    Extract data from this medical document. The document may contain MULTIPLE patients/records in a list or table.
    REQUIRED EXACT KEYS: [{columns}]
    USER RULES: {rules}
    
    STRICT JSON PROTOCOL:
    1. Output EXACTLY ONE valid JSON ARRAY containing objects. Format: [{{...}}, {{...}}]
    2. Create one JSON object inside the array for EACH patient/row found on the page.
    3. The keys MUST exactly match the REQUIRED KEYS above. 
    4. If a value is missing for a specific patient, output "N/A".
    5. Output ONLY the raw JSON array. No markdown, no conversational text.
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
    )
    
    raw_output = response.choices[0].message.content.strip()
    # Robust Regex to extract JSON Array even if the AI hallucinates text
    match = re.search(r'\[.*\]', raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return raw_output

# --- 5. SYSTEMATIC SIDEBAR ---
with st.sidebar:
    st.header("📋 1. Define Schema")
    st.warning("Do NOT type 'System_ID' here. The app generates it automatically.")
    column_headers = st.text_input("Exact Clinical Columns:", "Age, Gender, Organism, Ciprofloxacin")
    extra_rules = st.text_area("Specific Rules:", "If marked S write Sensitive. If R write Resistant. Strip colons.")
    
    st.divider()
    st.header("🔗 2. Connect Your Database")
    st.info("Share your Google Sheet as an 'Editor' with:\n\n**research-bot@dotted-ranger-490614-n9.iam.gserviceaccount.com**")
    user_sheet_url = st.text_input("Paste your Google Sheet URL here:")
    project_tab = st.text_input("Project Tab Name:", "Red_Eye_Study")
    
    st.divider()
    st.header("💾 3. Local Controls")
    unique_patients = st.session_state.master_database['System_ID'].nunique() if not st.session_state.master_database.empty else 0
    st.metric(label="Patients in Local App", value=unique_patients)
    
    if st.button("🚨 Clear Local Memory"):
        st.session_state.master_database = pd.DataFrame()
        st.rerun()

# --- 6. MAIN WORKFLOW: THE DUAL-MODE ENGINE ---
st.subheader("📸 Step 1: Data Entry Mode")

entry_mode = st.radio(
    "Select your workflow:",
    ["🆕 Add New Patients (App generates unique IDs automatically)", "🔄 Update Existing Patient (Add new lab data to an old profile)"],
    index=0
)

st.divider()
expected_cols = [c.strip() for c in column_headers.split(',')]

if "Add New" in entry_mode:
    st.info("Upload rosters or reports. The app will extract all rows and assign a unique 'System_ID' (e.g., CR-1001) to each.")
    with st.form("add_new_form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload patient documents:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        submitted = st.form_submit_button("⚙️ Extract & Generate IDs", type="primary")

    if submitted and uploaded_files:
        patient_dfs = []
        for file in uploaded_files:
            with st.spinner(f"AI is reading {file.name}..."):
                try:
                    raw_json = blueprint_decoder(file.getvalue(), column_headers, extra_rules)
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
            
            # THE AUTO-ID GENERATOR
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
    st.info("Look at your Google Sheet/Table below to find the patient's 'System_ID'. Type it to append new data.")
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
                        raw_json = blueprint_decoder(file.getvalue(), column_headers, extra_rules)
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

# --- 7. HUMAN-IN-THE-LOOP EDITOR ---
if not st.session_state.master_database.empty:
    st.divider()
    st.subheader("📝 Step 2: Verify & Edit Data")
    st.info("Double-click any cell to manually correct AI mistakes before syncing.")
    
    st.session_state.master_database = st.data_editor(
        st.session_state.master_database, 
        num_rows="dynamic", 
        use_container_width=True,
        key="data_verifier"
    )

# --- 8. LIVE RESEARCH DASHBOARD (UNIVERSAL) ---
if not st.session_state.master_database.empty:
    st.divider()
    st.subheader("📈 Step 3: Dynamic Data Explorer")
    
    all_cols = [c for c in st.session_state.master_database.columns if c != "System_ID"]
    
    tab1, tab2 = st.tabs(["📊 Single Variable", "🔄 Cross-Analysis"])
    
    with tab1:
        col_x, col_type = st.columns([2, 1])
        with col_x:
            var_1 = st.selectbox("Select variable to analyze:", [""] + all_cols, key="var1")
        with col_type:
            chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"])
            
        if var_1:
            df_clean = st.session_state.master_database[st.session_state.master_database[var_1] != 'N/A']
            val_counts = df_clean[var_1].value_counts().reset_index()
            val_counts.columns = [var_1, 'Count']
            
            if not val_counts.empty:
                if chart_type == "Bar Chart":
                    st.plotly_chart(px.bar(val_counts, x=var_1, y='Count', color=var_1), use_container_width=True)
                else:
                    st.plotly_chart(px.pie(val_counts, names=var_1, values='Count', hole=0.3), use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            x_axis = st.selectbox("X-Axis (Primary Group):", [""] + all_cols, key="x_axis")
        with col_b:
            y_axis = st.selectbox("Color Grouping (Sub-category):", [""] + all_cols, key="y_axis")
            
        if x_axis and y_axis:
            df_clean2 = st.session_state.master_database[(st.session_state.master_database[x_axis] != 'N/A') & (st.session_state.master_database[y_axis] != 'N/A')]
            if not df_clean2.empty:
                compare_counts = df_clean2.groupby([x_axis, y_axis]).size().reset_index(name='Count')
                st.plotly_chart(px.bar(compare_counts, x=x_axis, y='Count', color=y_axis, barmode='group'), use_container_width=True)

# --- 9. CLOUD SYNC ---
st.divider()
st.subheader("🌐 Step 4: Finalize & Sync")

col_x, col_y = st.columns([1, 1])

with col_x:
    if st.button("🚀 PUSH / PULL GOOGLE CLOUD", type="primary", use_container_width=True):
        with st.spinner("Syncing secure data..."):
            if not user_sheet_url or not project_tab:
                st.error("❌ Please fill out the Google Sheet URL and Tab Name in the sidebar!")
            else:
                try:
                    merged_data = sync_with_google_sheets(st.session_state.master_database, user_sheet_url, project_tab)
                    st.session_state.master_database = merged_data
                    st.success(f"✅ Sync Complete! Data secured in '{project_tab}'.")
                except Exception as e:
                    st.error(f"❌ Sync Failed. Error: {e}")

with col_y:
    if not st.session_state.master_database.empty:
        csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Offline CSV Backup", data=csv_data, file_name="Clinical_Data.csv", mime="text/csv", use_container_width=True)
