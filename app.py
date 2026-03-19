import streamlit as st
import pandas as pd
import numpy as np
import base64
import json
from groq import Groq
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image
import io

# --- 1. UI SETUP ---
st.set_page_config(page_title="CloudResearch AI", layout="wide", page_icon="☁️")
st.title("☁️ CloudResearch Patient Tracker")

# --- 2. THE MASTER DATABASE (MEMORY) ---
if "master_database" not in st.session_state:
    st.session_state.master_database = pd.DataFrame()

# --- 3. API SETUPS (CLOUD VAULT INTEGRATION) ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=GROQ_API_KEY)

GOOGLE_SHEET_NAME = "CloudResearch Live Database"

# 🔄 TWO-WAY SYNC ENGINE
def sync_with_google_sheets(local_dataframe, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    
    google_client = gspread.authorize(creds)
    sheet = google_client.open(sheet_name).sheet1
    
    cloud_data = sheet.get_all_values()
    if len(cloud_data) > 1:
        cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
    elif len(cloud_data) == 1:
        cloud_df = pd.DataFrame(columns=cloud_data[0])
    else:
        cloud_df = pd.DataFrame()
        
    if not cloud_df.empty:
        cloud_df.rename(columns=lambda x: 'Patient_ID' if str(x).strip().lower() in ['patient id', 'patient_id', 'patient-id'] else x, inplace=True)
    if not local_dataframe.empty:
        local_dataframe.rename(columns=lambda x: 'Patient_ID' if str(x).strip().lower() in ['patient id', 'patient_id', 'patient-id'] else x, inplace=True)

    if not local_dataframe.empty:
        combined_df = pd.concat([local_dataframe, cloud_df], ignore_index=True)
        combined_df['Patient_ID'] = combined_df['Patient_ID'].astype(str).str.strip()
        combined_df = combined_df[combined_df['Patient_ID'] != '']
        combined_df = combined_df[combined_df['Patient_ID'] != 'nan']
        combined_df.replace({'N/A': np.nan, 'NA': np.nan, '': np.nan, 'None': np.nan}, inplace=True)
        combined_df = combined_df.groupby('Patient_ID', as_index=False).first()
        combined_df.fillna('N/A', inplace=True)
    else:
        combined_df = cloud_df
        
    sheet.clear()
    if not combined_df.empty:
        combined_df = combined_df.astype(str)
        data_to_upload = [combined_df.columns.values.tolist()] + combined_df.values.tolist()
        sheet.update(range_name="A1", values=data_to_upload)
    
    return combined_df

# --- 4. THE BLUEPRINT DECODER (NOW WITH COMPRESSOR) ---
def compress_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((1024, 1024)) # Shrink massive phone photos
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()

def blueprint_decoder(image_bytes, columns, rules):
    # 1. Compress the raw upload first
    compressed_bytes = compress_image(image_bytes)
    base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
    
    prompt = f"""
    Extract data from this medical document.
    REQUIRED EXACT KEYS: [{columns}]
    USER RULES: {rules}
    
    STRICT JSON PROTOCOL:
    1. Output EXACTLY ONE valid JSON object.
    2. The keys MUST exactly match the REQUIRED KEYS above. 
    3. Do NOT add colons, spaces, or extra characters to the key names.
    4. If a value is missing, output the string "N/A".
    5. Output ONLY the raw JSON format with curly braces {{}}. No markdown.
    """
    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
    )
    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        parts = raw_output.split("```")
        raw_output = parts[1].replace("json", "").strip() if len(parts) > 1 else raw_output
    return raw_output

# --- 5. SYSTEMATIC SIDEBAR ---
with st.sidebar:
    st.header("📋 1. Define Schema")
    st.warning("Do not include 'Patient ID' here. The app handles it.")
    column_headers = st.text_input("Exact Clinical Columns:", "Age, Gender, ALT, AST")
    extra_rules = st.text_area("Specific Rules:", "If a value is not on the page, write N/A. Strip colons.")
    
    st.divider()
    st.header("💾 2. Database Controls")
    if not st.session_state.master_database.empty:
        unique_patients = st.session_state.master_database['Patient_ID'].nunique()
    else:
        unique_patients = 0
    st.metric(label="Patients in Local App", value=unique_patients)
    
    if st.button("🚨 Clear Local Memory"):
        st.session_state.master_database = pd.DataFrame()
        st.rerun()

# --- 6. MAIN WORKFLOW: THE SELF-CLEANING FORM ---
st.subheader("👤 Step 1: Process Current Patient")

with st.form("patient_processing_form", clear_on_submit=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        current_patient_id = st.text_input("Enter Patient ID / Initials:", placeholder="e.g., P-101")
    with col2:
        uploaded_files = st.file_uploader("Upload pages for this patient", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    submitted = st.form_submit_button("⚙️ Extract Data & Save to Local Database", type="primary")

if submitted:
    if not current_patient_id:
        st.error("Please enter a Patient ID first!")
    elif not uploaded_files:
        st.error("Please upload at least one image!")
    else:
        patient_dfs = []
        expected_cols = [c.strip() for c in column_headers.split(',')]
        
        for file in uploaded_files:
            with st.spinner(f"AI is reading {file.name}..."):
                try:
                    raw_json = blueprint_decoder(file.getvalue(), column_headers, extra_rules)
                    ai_data = json.loads(raw_json)
                    filtered_data = {col: ai_data.get(col, ai_data.get(f"{col}:", 'N/A')) for col in expected_cols}
                    df = pd.DataFrame([filtered_data])
                    patient_dfs.append(df)
                except Exception as e:
                    st.error(f"Could not read {file.name}. Ensure it is a clear image. Error details: {e}")
        
        if patient_dfs:
            current_batch_df = pd.concat(patient_dfs, ignore_index=True)
            current_batch_df.insert(0, "Patient_ID", str(current_patient_id).strip())
            
            st.session_state.master_database = pd.concat([st.session_state.master_database, current_batch_df], ignore_index=True)
            st.session_state.master_database.replace({'N/A': np.nan, 'NA': np.nan, '': np.nan, 'None': np.nan}, inplace=True)
            st.session_state.master_database = st.session_state.master_database.groupby('Patient_ID', as_index=False).first()
            st.session_state.master_database.fillna('N/A', inplace=True)
            
            st.success(f"✅ Patient {current_patient_id} locally processed!")

# --- 7. VIEW & SYNC TO CLOUD ---
st.divider()
st.subheader("📊 Step 2: Live Cloud Synchronization")

col_x, col_y = st.columns([1, 1])

with col_x:
    if st.button("🌐 TWO-WAY SYNC (Pull & Push Google Sheets)", type="primary", use_container_width=True):
        with st.spinner("Connecting to Google Cloud..."):
            try:
                merged_data = sync_with_google_sheets(st.session_state.master_database, GOOGLE_SHEET_NAME)
                st.session_state.master_database = merged_data
                st.success("✅ Sync Complete! The cloud and your app are now identical.")
            except Exception as e:
                st.error(f"❌ Sync Failed. Check credentials.json and sharing settings. Error: {e}")

with col_y:
    if not st.session_state.master_database.empty:
        csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Offline CSV", data=csv_data, file_name="Offline_Research.csv", mime="text/csv", use_container_width=True)

if not st.session_state.master_database.empty:
    st.dataframe(st.session_state.master_database, use_container_width=True)
else:
    st.info("The database is currently empty. Process a patient or click Sync to download existing cloud data.")
