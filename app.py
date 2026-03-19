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
import plotly.express as px

# --- 1. UI SETUP ---
st.set_page_config(page_title="CloudResearch AI", layout="wide", page_icon="☁️")
st.title("☁️ CloudResearch Command Center")

# --- 2. THE MASTER DATABASE (MEMORY) ---
if "master_database" not in st.session_state:
    st.session_state.master_database = pd.DataFrame()

# --- 3. API SETUPS (CLOUD VAULT INTEGRATION) ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=GROQ_API_KEY)

# 🔄 TWO-WAY SYNC ENGINE (Multi-User & Multi-Project)
def sync_with_google_sheets(local_dataframe, sheet_url, tab_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    
    google_client = gspread.authorize(creds)
    sheet = google_client.open_by_url(sheet_url).worksheet(tab_name)
    
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
        # Keeps the most recently updated version of a patient's data
        combined_df = combined_df.groupby('Patient_ID', as_index=False).last()
        combined_df.fillna('N/A', inplace=True)
    else:
        combined_df = cloud_df
        
    sheet.clear()
    if not combined_df.empty:
        combined_df = combined_df.astype(str)
        data_to_upload = [combined_df.columns.values.tolist()] + combined_df.values.tolist()
        sheet.update(range_name="A1", values=data_to_upload)
    
    return combined_df

# --- 4. THE BLUEPRINT DECODER (HIGH-RES) ---
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
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
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
    column_headers = st.text_input("Exact Clinical Columns:", "Organism, Ciprofloxacin, Amoxicillin")
    extra_rules = st.text_area("Specific Rules:", "If marked S write Sensitive. If R write Resistant. Strip colons.")
    
    st.divider()
    st.header("🔗 2. Connect Your Database")
    st.info("Share your Google Sheet as an 'Editor' with:\n\n**research-bot@dotted-ranger-490614-n9.iam.gserviceaccount.com**")
    user_sheet_url = st.text_input("Paste your Google Sheet URL here:")
    project_tab = st.text_input("Project Tab Name (Exact Match):", "Sheet1")
    
    st.divider()
    st.header("💾 3. Local Controls")
    if not st.session_state.master_database.empty:
        unique_patients = st.session_state.master_database['Patient_ID'].nunique()
    else:
        unique_patients = 0
    st.metric(label="Patients in Local App", value=unique_patients)
    
    if st.button("🚨 Clear Local Memory"):
        st.session_state.master_database = pd.DataFrame()
        st.rerun()

# --- 6. MAIN WORKFLOW: DATA EXTRACTION ---
st.subheader("📸 Step 1: Extract Lab Reports")

with st.form("patient_processing_form", clear_on_submit=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        current_patient_id = st.text_input("Enter Patient ID:", placeholder="e.g., P-101")
    with col2:
        uploaded_files = st.file_uploader("Upload pages for this patient", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    submitted = st.form_submit_button("⚙️ Run AI Extraction", type="primary")

if submitted:
    if not current_patient_id or not uploaded_files:
        st.error("Please enter a Patient ID and upload an image!")
    else:
        patient_dfs = []
        expected_cols = [c.strip() for c in column_headers.split(',')]
        
        for file in uploaded_files:
            with st.spinner(f"AI is reading {file.name}..."):
                try:
                    raw_json = blueprint_decoder(file.getvalue(), column_headers, extra_rules)
                    ai_data = json.loads(raw_json)
                    
                    filtered_data = {col: ai_data.get(col, ai_data.get(f"{col}:", 'N/A')) for col in expected_cols}
                    patient_dfs.append(pd.DataFrame([filtered_data]))
                except Exception as e:
                    st.error(f"Could not read {file.name}. Error: {e}")
        
        if patient_dfs:
            current_batch_df = pd.concat(patient_dfs, ignore_index=True)
            current_batch_df.insert(0, "Patient_ID", str(current_patient_id).strip())
            
            st.session_state.master_database = pd.concat([st.session_state.master_database, current_batch_df], ignore_index=True)
            st.session_state.master_database.replace({'N/A': np.nan, 'NA': np.nan, '': np.nan, 'None': np.nan}, inplace=True)
            st.session_state.master_database = st.session_state.master_database.groupby('Patient_ID', as_index=False).last()
            st.session_state.master_database.fillna('N/A', inplace=True)
            st.success(f"✅ Patient {current_patient_id} processed!")

# --- 7. HUMAN-IN-THE-LOOP EDITOR ---
if not st.session_state.master_database.empty:
    st.divider()
    st.subheader("📝 Step 2: Verify & Edit Data")
    st.info("Double-click any cell below to manually correct AI mistakes before syncing.")
    
    # The interactive data editor overrides the session state
    st.session_state.master_database = st.data_editor(
        st.session_state.master_database, 
        num_rows="dynamic", 
        use_container_width=True,
        key="data_verifier"
    )

# --- 8. LIVE RESEARCH DASHBOARD (UNIVERSAL UPGRADE) ---
if not st.session_state.master_database.empty:
    st.divider()
    st.subheader("📈 Step 3: Dynamic Data Explorer")
    st.info("Visualize your data instantly. The charts automatically adapt to your custom clinical columns.")
    
    # Get all columns except the Patient ID
    all_cols = [c for c in st.session_state.master_database.columns if c != "Patient_ID"]
    
    tab1, tab2 = st.tabs(["📊 Distribution (Single Variable)", "🔄 Cross-Analysis (Two Variables)"])
    
    with tab1:
        col_x, col_type = st.columns([2, 1])
        with col_x:
            var_1 = st.selectbox("Select variable to analyze:", [""] + all_cols, key="var1")
        with col_type:
            chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"])
            
        if var_1:
            # Clean out N/A values for graphing
            df_clean = st.session_state.master_database[st.session_state.master_database[var_1] != 'N/A']
            val_counts = df_clean[var_1].value_counts().reset_index()
            val_counts.columns = [var_1, 'Count']
            
            if not val_counts.empty:
                if chart_type == "Bar Chart":
                    fig1 = px.bar(val_counts, x=var_1, y='Count', color=var_1, title=f"Distribution of {var_1}")
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    fig1 = px.pie(val_counts, names=var_1, values='Count', title=f"Breakdown of {var_1}", hole=0.3)
                    st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning(f"No valid data to plot for {var_1} yet.")

    with tab2:
        st.write("Compare how one variable interacts with another (e.g., Flora vs. Antibiotic Resistance).")
        col_a, col_b = st.columns(2)
        with col_a:
            x_axis = st.selectbox("X-Axis (Primary Group):", [""] + all_cols, key="x_axis")
        with col_b:
            y_axis = st.selectbox("Color Grouping (Sub-category):", [""] + all_cols, key="y_axis")
            
        if x_axis and y_axis:
            df_clean2 = st.session_state.master_database[(st.session_state.master_database[x_axis] != 'N/A') & (st.session_state.master_database[y_axis] != 'N/A')]
            
            if not df_clean2.empty:
                compare_counts = df_clean2.groupby([x_axis, y_axis]).size().reset_index(name='Count')
                fig2 = px.bar(compare_counts, x=x_axis, y='Count', color=y_axis, barmode='group', title=f"{y_axis} grouped by {x_axis}")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Not enough overlapping data to compare these two variables yet.")

# --- 9. CLOUD SYNC ---
st.divider()
st.subheader("🌐 Step 4: Finalize & Sync")

col_x, col_y = st.columns([1, 1])

with col_x:
    if st.button("🚀 PUSH TO GOOGLE CLOUD", type="primary", use_container_width=True):
        with st.spinner("Syncing secure data..."):
            if not user_sheet_url or not project_tab:
                st.error("❌ Please fill out the Google Sheet URL and Tab Name in the sidebar!")
            else:
                try:
                    merged_data = sync_with_google_sheets(st.session_state.master_database, user_sheet_url, project_tab)
                    st.session_state.master_database = merged_data
                    st.success(f"✅ Sync Complete! Data secured in '{project_tab}'.")
                except Exception as e:
                    st.error(f"❌ Sync Failed. Check URL and Tab name. Error: {e}")

with col_y:
    if not st.session_state.master_database.empty:
        csv_data = st.session_state.master_database.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Offline CSV Backup", data=csv_data, file_name="Clinical_Data.csv", mime="text/csv", use_container_width=True)
