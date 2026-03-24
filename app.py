import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import base64
import json
import re
import random
import string
import time
import logging
from datetime import datetime
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image
import fitz
import io
import plotly.express as px

# ═══════════════════════════════════════════════════════════════
# 0. CONSTANTS & DEFAULTS
# ═══════════════════════════════════════════════════════════════

INVALID_VALUES = {'n/a', 'nan', '', 'none', 'null', 'na', 'n.a.', 'not available', 'not applicable'}

NUMERIC_KEYWORDS = [
    'age', 'weight', 'height', 'dose', 'duration', 'count', 'level',
    'score', 'year', 'days', 'weeks', 'months', 'wbc', 'crp', 'esr',
    'hb', 'plt', 'mic', 'los', 'temperature', 'bp', 'hr', 'rr', 'spo2'
]

DEFAULT_PROMPT = (
    "You are an expert clinical data extraction engine. "
    "Extract all structured medical and demographic data from the provided document. "
    "Apply standard clinical abbreviation expansion. "
    "Return ONLY a valid JSON array — no explanation, no preamble, no markdown."
)

DEFAULT_BUILT_IN_RULES = (
    "Expand abbreviations (M=Male, F=Female, HTN=Hypertension, DM=Diabetes Mellitus). "
    "Standardize units where present. "
    "Use exactly the string 'N/A' for any missing or unreadable field. "
    "Never fabricate or infer data not explicitly present in the document."
)

PROMPT_TEMPLATES = {
    "— Select a Template —": {
        "schema": "", "prompt": "", "abbreviations": "", "rules": "", "anti": ""
    },
    "🦠 Microbiology": {
        "schema": "Patient_ID, Age, Gender, Organism, Antibiotic, MIC, Resistance_Pattern, Specimen_Type, Culture_Date, Outcome",
        "prompt": "Extract microbiological culture and sensitivity data. Focus on organism identity, antibiotic susceptibility, and clinical outcome.",
        "abbreviations": "MIC=Minimum Inhibitory Concentration, MDR=Multi-Drug Resistant, MRSA=Methicillin-Resistant S. aureus, S=Sensitive, R=Resistant, I=Intermediate, ESBL=Extended-Spectrum Beta-Lactamase",
        "rules": "Record each organism–antibiotic pair as a separate entry. If multiple organisms, create separate entries for each.",
        "anti": "Exclude administrative billing codes and non-microbiological findings."
    },
    "👤 Demographics": {
        "schema": "Patient_ID, Age, Gender, Nationality, Comorbidities, Chief_Complaint, Admission_Date, Discharge_Date, LOS_Days",
        "prompt": "Extract patient demographic and admission data from clinical records.",
        "abbreviations": "LOS=Length of Stay, HTN=Hypertension, DM=Diabetes Mellitus, CAD=Coronary Artery Disease, CKD=Chronic Kidney Disease, CVA=Cerebrovascular Accident",
        "rules": "Calculate LOS_Days if both admission and discharge dates are present. List comorbidities as comma-separated values.",
        "anti": "Exclude laboratory results and medication details from this schema."
    },
    "🧪 Laboratory": {
        "schema": "Patient_ID, Test_Name, Result, Unit, Reference_Range, Flag, Collection_Date, Interpretation",
        "prompt": "Extract all laboratory investigation results with units, reference ranges, and clinical flags.",
        "abbreviations": "WBC=White Blood Cells, Hb=Hemoglobin, PLT=Platelets, CRP=C-Reactive Protein, ESR=Erythrocyte Sedimentation Rate, HbA1c=Glycated Hemoglobin, eGFR=Estimated Glomerular Filtration Rate",
        "rules": "Flag values outside reference range as HIGH or LOW. Extract each test as a separate JSON object.",
        "anti": "Exclude demographic information from laboratory entries."
    },
    "💊 Pharmacology": {
        "schema": "Patient_ID, Drug_Name, Dose, Route, Frequency, Duration_Days, Indication, ADR, Outcome",
        "prompt": "Extract medication administration records including dosing, route, and any adverse drug reactions.",
        "abbreviations": "IV=Intravenous, PO=Per Oral, BD=Twice Daily, TDS=Three Times Daily, OD=Once Daily, ADR=Adverse Drug Reaction, PRN=As Needed, SC=Subcutaneous",
        "rules": "Record each drug as a separate entry. Note any dose modifications or discontinuations explicitly.",
        "anti": "Exclude non-pharmaceutical interventions and procedure notes."
    },
}

# ═══════════════════════════════════════════════════════════════
# 1. LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloudresearch")

# ═══════════════════════════════════════════════════════════════
# 2. UI SETUP & GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="CloudResearch Command Center", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.markdown("""
    <style>
        h1  { font-size: 1.5rem  !important; font-weight: 700 !important; padding-bottom: 0.5rem !important; }
        h2  { font-size: 1.1rem  !important; font-weight: 600 !important; padding-top: 1rem   !important; padding-bottom: 0.2rem !important; }
        h3  { font-size: 1.05rem !important; font-weight: 600 !important; padding-bottom: 0.2rem !important; }
        .streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.95rem !important; }
        .stAlert { border-radius: 6px !important; }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 3. HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════

def is_valid_value(value: str) -> bool:
    """Return True if value contains real data (not a placeholder)."""
    return str(value).strip().lower() not in INVALID_VALUES


def generate_unique_id(existing_ids: set) -> str:
    """Generate a collision-free CR-XXXX identifier."""
    alphabet = string.ascii_uppercase + string.digits
    while True:
        candidate = "CR-" + "".join(random.choices(alphabet, k=4))
        if candidate not in existing_ids:
            return candidate


def parse_ai_json_safe(raw_text: str) -> tuple[list[dict], str | None]:
    """
    Safely parse AI output into a list of dicts.
    Returns (records_list, error_message_or_None).
    Never raises; always returns a (possibly empty) list.
    """
    if not raw_text or not raw_text.strip():
        return [], "AI returned an empty response."

    # Strip markdown code fences
    cleaned = raw_text.strip()
    for fence in ["```json", "```"]:
        if fence in cleaned:
            parts = cleaned.split(fence)
            cleaned = parts[1] if len(parts) > 1 else cleaned

    # Extract the first JSON array or object
    match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
    if not match:
        return [], f"No valid JSON structure found in output. Raw snippet: {cleaned[:120]}"

    extracted = match.group(1).strip()

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode failed: %s | raw: %s", exc, extracted[:200])
        return [], f"JSON parse error: {exc}"

    if isinstance(parsed, dict):
        # OpenAI wraps in {"data": [...]}
        for key in ("data", "records", "patients", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key], None
        return [parsed], None

    if isinstance(parsed, list):
        return parsed, None

    return [], f"Unexpected JSON type: {type(parsed).__name__}"


def normalize_record(record: dict, expected_cols: list[str]) -> dict:
    """
    Ensure a record contains exactly the expected columns.
    Missing columns are filled with 'N/A'. Extra columns are discarded.
    Also tries fuzzy key matching (strips trailing colons, case-insensitive).
    """
    # Build a case-insensitive lookup of the raw record
    lower_map = {k.lower().rstrip(":"): v for k, v in record.items()}

    normalized = {}
    for col in expected_cols:
        col_key = col.lower().strip()
        value = lower_map.get(col_key, "N/A")
        normalized[col] = str(value).strip() if is_valid_value(str(value)) else "N/A"
    return normalized


def validate_and_clean_dataframe(df: pd.DataFrame, expected_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate a DataFrame against the expected schema.
    Returns (cleaned_df, list_of_warning_messages).
    """
    warnings: list[str] = []

    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "N/A"
            warnings.append(f"Column '{col}' was missing and has been added with default values.")

    # Attempt numeric coercion on columns whose names contain numeric keywords
    for col in df.columns:
        if any(kw in col.lower() for kw in NUMERIC_KEYWORDS):
            original = df[col].copy()
            coerced = pd.to_numeric(df[col].replace("N/A", pd.NA), errors="coerce")
            failed = coerced.isna() & original.notna() & (original != "N/A")
            if failed.any():
                warnings.append(
                    f"Column '{col}': {failed.sum()} non-numeric value(s) left as-is — "
                    f"e.g. '{original[failed].iloc[0]}'."
                )

    # Standardise placeholder variants across all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
            {v: "N/A" for v in {"nan", "None", "NaN", "none", "null", "NULL", "N/A", "NA"}},
            regex=False
        )

    return df, warnings


def add_audit_columns(df: pd.DataFrame, username: str) -> pd.DataFrame:
    """Append Created_By and Timestamp columns if they don't already exist."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if "Created_By" not in df.columns:
        df["Created_By"] = username
    if "Timestamp" not in df.columns:
        df["Timestamp"] = ts
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows by comparing all columns except System_ID, Timestamp, Created_By.
    Returns (deduplicated_df, number_of_rows_removed).
    """
    if df.empty:
        return df, 0
    key_cols = [c for c in df.columns if c not in {"System_ID", "Timestamp", "Created_By"}]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return df, before - len(df)

# ═══════════════════════════════════════════════════════════════
# 4. GOOGLE SHEETS INTEGRATION
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def get_google_sheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)


def _get_or_create_worksheet(sheet_url: str, tab_name: str):
    """Open worksheet, creating it if absent."""
    client = get_google_sheet_client()
    try:
        return client.open_by_url(sheet_url).worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = client.open_by_url(sheet_url)
        return spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="30")


def pull_from_sheet(sheet_url: str, tab_name: str) -> pd.DataFrame:
    """Pull data from Google Sheets into a DataFrame, auto-assigning missing System_IDs."""
    sheet = _get_or_create_worksheet(sheet_url, tab_name)
    cloud_data = sheet.get_all_values()

    if len(cloud_data) > 1:
        cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
    elif len(cloud_data) == 1:
        cloud_df = pd.DataFrame(columns=cloud_data[0])
    else:
        return pd.DataFrame()

    if cloud_df.empty:
        return cloud_df

    # Normalise System_ID column name
    cloud_df.rename(
        columns=lambda x: "System_ID" if str(x).strip().lower() in {"system_id", "system id"} else x,
        inplace=True
    )

    if "System_ID" not in cloud_df.columns:
        existing: set[str] = set()
        new_ids = [generate_unique_id(existing := existing | {nid}) or nid
                   for _ in range(len(cloud_df))
                   for nid in [generate_unique_id(existing)]]
        # Simpler:
        existing = set()
        new_ids = []
        for _ in range(len(cloud_df)):
            nid = generate_unique_id(existing)
            new_ids.append(nid)
            existing.add(nid)
        cloud_df.insert(0, "System_ID", new_ids)
    else:
        missing_mask = cloud_df["System_ID"].astype(str).str.strip().str.lower().isin(INVALID_VALUES)
        if missing_mask.any():
            existing = set(cloud_df.loc[~missing_mask, "System_ID"].astype(str).tolist())
            new_ids = []
            for _ in range(missing_mask.sum()):
                nid = generate_unique_id(existing)
                new_ids.append(nid)
                existing.add(nid)
            cloud_df.loc[missing_mask, "System_ID"] = new_ids

    return cloud_df


def push_to_sheet_merge(local_df: pd.DataFrame, sheet_url: str, tab_name: str) -> pd.DataFrame:
    """
    Safe merge-push strategy:
    1. Pull existing cloud data.
    2. Remove cloud rows whose System_ID is present in local_df (local takes priority).
    3. Concatenate cloud remainder + local, then push.
    This prevents total data loss if the local cache was incomplete.
    """
    sheet = _get_or_create_worksheet(sheet_url, tab_name)

    try:
        cloud_df = pull_from_sheet(sheet_url, tab_name)
    except Exception as exc:
        logger.warning("Could not pull existing cloud data before push: %s", exc)
        cloud_df = pd.DataFrame()

    if (
        not cloud_df.empty
        and "System_ID" in cloud_df.columns
        and not local_df.empty
        and "System_ID" in local_df.columns
    ):
        preserved = cloud_df[~cloud_df["System_ID"].isin(local_df["System_ID"])]
        merged_df = pd.concat([preserved, local_df], ignore_index=True)
    else:
        merged_df = local_df.copy()

    merged_df = merged_df.astype(str)
    cols = ["System_ID"] + [c for c in merged_df.columns if c != "System_ID"]
    merged_df = merged_df[[c for c in cols if c in merged_df.columns]]

    sheet.clear()
    if not merged_df.empty:
        sheet.update(range_name="A1", values=[merged_df.columns.tolist()] + merged_df.values.tolist())
    else:
        sheet.update(range_name="A1", values=[["System_ID"]])

    return merged_df

# ═══════════════════════════════════════════════════════════════
# 5. PROFILE SYNC (ADMIN DIRECTORY)
# ═══════════════════════════════════════════════════════════════

_PROFILE_COLUMNS = ["Username", "Target_Sheet_URL", "Schema", "Prompt", "Abbreviations", "Extra_Rules", "Anti_Rules"]


def sync_user_profile(username: str, mode: str = "pull", profile_data: dict | None = None):
    admin_url = st.secrets.get("ADMIN_SHEET_URL", "")
    if not admin_url:
        return None

    client = get_google_sheet_client()
    try:
        sheet = client.open_by_url(admin_url).worksheet("Profiles")
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = client.open_by_url(admin_url)
        sheet = spreadsheet.add_worksheet(title="Profiles", rows="100", cols="10")
        sheet.update(range_name="A1", values=[_PROFILE_COLUMNS])

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
            profile_data.get("Anti_Rules", ""),
        ]
        if row_idx:
            sheet.update(range_name=f"A{row_idx}:G{row_idx}", values=[row_values])
        else:
            sheet.append_row(row_values)
        return True

# ═══════════════════════════════════════════════════════════════
# 6. IMAGE / PDF PROCESSING
# ═══════════════════════════════════════════════════════════════

def compress_image(image_bytes: bytes) -> bytes:
    """Grayscale + resize + JPEG compress to reduce API token usage."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L")
    img.thumbnail((768, 768))
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=75)
    return output.getvalue()


def convert_pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Rasterise each PDF page to a JPEG byte string."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        images.append(pix.tobytes("jpeg"))
    return images

# ═══════════════════════════════════════════════════════════════
# 7. PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_final_prompt(user_prompt: str, abbreviations: str, extra_rules: str, anti_rules: str) -> str:
    """
    Assemble a minified, token-efficient extraction prompt.
    Injects default rules when user leaves fields blank.
    """
    effective_prompt = user_prompt.strip() or DEFAULT_PROMPT
    effective_rules  = extra_rules.strip()  or DEFAULT_BUILT_IN_RULES
    abbrev_block     = f"Map: {abbreviations.strip()}." if abbreviations.strip() else ""
    anti_block       = f"Avoid: {anti_rules.strip()}."  if anti_rules.strip()  else ""

    return (
        f"Task: Extract clinical data to JSON array. "
        f"Directive: {effective_prompt}. "
        f"{abbrev_block} "
        f"Rules: {effective_rules}. "
        f"{anti_block} "
        f"Output RAW JSON array `[{{...}}]` ONLY. "
        f"No markdown. No explanation. Use 'N/A' for missing fields."
    ).strip()

# ═══════════════════════════════════════════════════════════════
# 8. AI EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════════

def blueprint_decoder(
    image_bytes: bytes,
    schema_columns: str,
    final_prompt: str,
    model_choice: str
) -> tuple[str, str | None]:
    """
    Send an image to the selected AI model and return the raw JSON string.
    Returns (raw_json_string, error_message_or_None).
    """
    full_prompt = (
        f"{final_prompt}\n\n"
        f"REQUIRED JSON KEYS: [{schema_columns}]\n\n"
        f"Output format: valid JSON ARRAY `[{{...}}]`. "
        f"Multiple patients → one object per patient. "
        f"Unreadable image → output empty array `[]`. "
        f"RETURN ONLY THE JSON. NO OTHER TEXT."
    )

    compressed = compress_image(image_bytes)
    b64_image = base64.b64encode(compressed).decode("utf-8")

    try:
        if "Gemini" in model_choice:
            img = Image.open(io.BytesIO(compressed))
            gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
            model = genai.GenerativeModel(gemini_model_name)
            response = model.generate_content(
                [full_prompt, img],
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text.strip(), None

        elif "Groq" in model_choice:
            client = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt + " ONLY output the raw JSON array."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }]
            )
            return response.choices[0].message.content.strip(), None

        elif "OpenAI" in model_choice:
            client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt + " Wrap the array in a JSON object with key 'data'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }]
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            return json.dumps(parsed.get("data", [])), None

    except Exception as exc:
        logger.error("Model error [%s]: %s", model_choice, exc)
        return "[]", f"{model_choice} error: {exc}"

    return "[]", "Unknown model selection."

# ═══════════════════════════════════════════════════════════════
# 9. AUTHENTICATION GATEKEEPER
# ═══════════════════════════════════════════════════════════════

def _deep_copy_dict(obj):
    """Recursively convert immutable mappings to plain dicts."""
    if isinstance(obj, dict) or hasattr(obj, "items"):
        return {k: _deep_copy_dict(v) for k, v in obj.items()}
    return obj


try:
    mutable_creds = _deep_copy_dict(st.secrets["credentials"])
    authenticator = stauth.Authenticate(
        mutable_creds,
        st.secrets["cookie"]["name"],
        st.secrets["cookie"]["key"],
        st.secrets["cookie"]["expiry_days"],
    )
    authenticator.login()
except Exception as exc:
    st.error("Authentication system failed to initialise.")
    st.error(f"Developer detail: {exc}")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# 10. MAIN APPLICATION (AUTH-GATED)
# ═══════════════════════════════════════════════════════════════

auth_status = st.session_state.get("authentication_status")

if auth_status is False:
    st.error("Incorrect username or password.")

elif auth_status is None:
    st.title("CloudResearch")
    st.warning("Enter your credentials to access the Command Center.")

elif auth_status is True:
    username: str = st.session_state.get("username", "")
    name: str     = st.session_state.get("name", "")

    # ── SESSION STATE INITIALISATION ────────────────────────────
    if "profile_loaded" not in st.session_state:
        with st.spinner("Loading secure profile…"):
            cloud_profile = sync_user_profile(username, mode="pull")
            if cloud_profile:
                st.session_state.target_sheet_url = cloud_profile.get("Target_Sheet_URL", "")
                st.session_state.schema_input     = cloud_profile.get("Schema", "Age, Gender, Organism")
                st.session_state.user_prompt      = cloud_profile.get("Prompt", DEFAULT_PROMPT)
                st.session_state.abbreviations    = cloud_profile.get("Abbreviations", "")
                st.session_state.extra_rules      = cloud_profile.get("Extra_Rules", "")
                st.session_state.anti_rules       = cloud_profile.get("Anti_Rules", "")
            else:
                st.session_state.target_sheet_url = ""
                st.session_state.schema_input     = "Age, Gender, Organism"
                st.session_state.user_prompt      = DEFAULT_PROMPT
                st.session_state.abbreviations    = ""
                st.session_state.extra_rules      = ""
                st.session_state.anti_rules       = ""
            st.session_state.profile_loaded = True

    if "master_database" not in st.session_state:
        st.session_state.master_database = pd.DataFrame()
        if st.session_state.target_sheet_url:
            try:
                st.session_state.master_database = pull_from_sheet(
                    st.session_state.target_sheet_url, username
                )
            except Exception as exc:
                logger.warning("Background sheet pull failed: %s", exc)

    if "safe_schema_val" not in st.session_state:
        st.session_state.safe_schema_val = st.session_state.schema_input

    # ── SIDEBAR ─────────────────────────────────────────────────
    with st.sidebar:
        st.success(f"✅ {name}")
        authenticator.logout("Logout", "sidebar")
        st.divider()

        # ── ⚙️ SETUP ──────────────────────────────────────────
        with st.expander("⚙️ Setup", expanded=True):
            selected_model = st.selectbox(
                "AI Model",
                ["Google Gemini (Primary)", "Groq (Free Fallback)", "OpenAI (Paid Fallback)"]
            )
            active_sheet_url = st.text_input("Google Sheet URL", value=st.session_state.target_sheet_url)
            st.session_state.target_sheet_url = active_sheet_url

        # ── 📋 SCHEMA ──────────────────────────────────────────
        with st.expander("📋 Schema", expanded=True):
            # Template selector
            tpl_choice = st.selectbox("Quick Template", list(PROMPT_TEMPLATES.keys()))
            if tpl_choice != "— Select a Template —":
                tpl = PROMPT_TEMPLATES[tpl_choice]
                if st.button("Apply Template", use_container_width=True):
                    st.session_state.safe_schema_val = tpl["schema"]
                    st.session_state.schema_input    = tpl["schema"]
                    st.session_state.user_prompt     = tpl["prompt"]
                    st.session_state.abbreviations   = tpl["abbreviations"]
                    st.session_state.extra_rules     = tpl["rules"]
                    st.session_state.anti_rules      = tpl["anti"]
                    st.rerun()

            col_pull, col_push = st.columns(2)
            with col_pull:
                if st.button("↓ Pull", use_container_width=True, help="Sync column headers from cloud sheet"):
                    if active_sheet_url:
                        with st.spinner("Pulling schema…"):
                            try:
                                sheet = get_google_sheet_client().open_by_url(active_sheet_url).worksheet(username)
                                headers = sheet.row_values(1)
                                clean   = [c for c in headers if c.lower() != "system_id"]
                                st.session_state.safe_schema_val = ", ".join(clean)
                                st.session_state.schema_input    = st.session_state.safe_schema_val
                                st.toast("Schema pulled.")
                                st.rerun()
                            except Exception as exc:
                                st.toast(f"Pull failed: {exc}")
            with col_push:
                if st.button("↑ Push", use_container_width=True, help="Write column headers to cloud sheet"):
                    if active_sheet_url:
                        with st.spinner("Pushing schema…"):
                            try:
                                sheet   = get_google_sheet_client().open_by_url(active_sheet_url).worksheet(username)
                                cols    = [c.strip() for c in st.session_state.safe_schema_val.split(",") if c.strip()]
                                headers = ["System_ID"] + [c for c in cols if c.lower() != "system_id"]
                                sheet.update(range_name="A1", values=[headers])
                                st.toast("Schema pushed.")
                            except Exception as exc:
                                st.toast(f"Push failed: {exc}")

            updated_schema = st.text_input("Columns (comma-separated)", value=st.session_state.safe_schema_val)
            st.session_state.safe_schema_val = updated_schema
            st.session_state.schema_input    = updated_schema

        # ── 🧠 EXTRACTION LOGIC ────────────────────────────────
        with st.expander("🧠 Extraction Logic"):
            st.session_state.user_prompt = st.text_area(
                "Primary Directive",
                value=st.session_state.user_prompt,
                height=120
            )
            st.session_state.abbreviations = st.text_area(
                "Abbreviations Map",
                value=st.session_state.abbreviations,
                height=80
            )
            st.session_state.extra_rules = st.text_area(
                "Inclusion Rules",
                value=st.session_state.extra_rules,
                height=80
            )
            st.session_state.anti_rules = st.text_area(
                "Exclusion Rules",
                value=st.session_state.anti_rules,
                height=60
            )

        # ── 💾 PROFILE ─────────────────────────────────────────
        with st.expander("💾 Profile"):
            if st.button("Save Configuration", type="primary", use_container_width=True):
                with st.spinner("Saving to central directory…"):
                    sync_user_profile(username, mode="push", profile_data={
                        "Target_Sheet_URL": st.session_state.target_sheet_url,
                        "Schema":           st.session_state.schema_input,
                        "Prompt":           st.session_state.user_prompt,
                        "Abbreviations":    st.session_state.abbreviations,
                        "Extra_Rules":      st.session_state.extra_rules,
                        "Anti_Rules":       st.session_state.anti_rules,
                    })
                    st.success("Profile saved.")

        # ── 🔧 SYSTEM ──────────────────────────────────────────
        with st.expander("🔧 System"):
            st.warning("Clears all unsaved local data.")
            if st.button("Purge Local Cache", use_container_width=True):
                st.session_state.master_database = pd.DataFrame()
                st.rerun()

    # ── MAIN CONTENT ─────────────────────────────────────────────
    st.title("CloudResearch Command Center")
    tabs = st.tabs(["📥 Data Entry & Sync", "🔍 Data Explorer"])
    expected_cols = [c.strip() for c in st.session_state.schema_input.split(",") if c.strip()]
    project_tab   = username

    # ═══════════════════════════════════════════════════════
    # TAB 1 — DATA ENTRY & SYNC
    # ═══════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Record Management")
        entry_mode = st.radio(
            "Processing Mode",
            ["Single Record (Compile Pages)", "Batch Processing (Roster Extract)", "Update Existing Record"],
            horizontal=True
        )
        st.divider()

        # ── SINGLE RECORD ─────────────────────────────────
        if "Single Record" in entry_mode:
            st.info("Upload all documents for one subject. The engine compiles them into a unified profile.")
            with st.form("single_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=["png", "jpg", "jpeg", "pdf"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(
                    f"Process via {selected_model.split(' ')[0]}", type="primary"
                )

            if submitted and uploaded_files:
                final_prompt = build_final_prompt(
                    st.session_state.user_prompt,
                    st.session_state.abbreviations,
                    st.session_state.extra_rules,
                    st.session_state.anti_rules
                )

                with st.spinner("Pre-processing documents…"):
                    ready_images: list[tuple[str, bytes]] = []
                    for f in uploaded_files:
                        if f.name.lower().endswith(".pdf"):
                            pages = convert_pdf_to_images(f.getvalue())
                            ready_images.extend((f"{f.name} p{i+1}", img) for i, img in enumerate(pages))
                        else:
                            ready_images.append((f.name, f.getvalue()))

                master_record = {col: "N/A" for col in expected_cols}
                processing_log: list[dict] = []

                progress = st.progress(0, text="Extracting…")
                for idx, (label, img_bytes) in enumerate(ready_images):
                    progress.progress((idx) / len(ready_images), text=f"Processing: {label}")
                    raw_json, api_error = blueprint_decoder(
                        img_bytes, st.session_state.schema_input, final_prompt, selected_model
                    )
                    time.sleep(4.5)  # Rate-limit buffer for Gemini free tier

                    if api_error:
                        processing_log.append({"file": label, "status": "❌ Failed", "records": 0, "detail": api_error})
                        continue

                    records, parse_error = parse_ai_json_safe(raw_json)
                    if parse_error or not records:
                        processing_log.append({"file": label, "status": "⚠️ No data", "records": 0, "detail": parse_error or "Empty output"})
                        continue

                    for rec in records:
                        for col in expected_cols:
                            new_val = str(rec.get(col, "N/A")).strip()
                            if is_valid_value(new_val) and not is_valid_value(master_record[col]):
                                pass  # keep existing
                            elif is_valid_value(new_val):
                                master_record[col] = new_val

                    processing_log.append({"file": label, "status": "✅ OK", "records": len(records), "detail": ""})

                progress.progress(1.0, text="Done.")

                # Build DataFrame, assign ID, add audit cols
                new_df = pd.DataFrame([master_record])
                new_df, warnings = validate_and_clean_dataframe(new_df, expected_cols)
                new_df = add_audit_columns(new_df, username)

                existing_ids = set(st.session_state.master_database.get("System_ID", pd.Series(dtype=str)).astype(str).tolist()) \
                    if not st.session_state.master_database.empty else set()
                new_id = generate_unique_id(existing_ids)
                new_df.insert(0, "System_ID", [new_id])

                st.session_state.master_database = pd.concat(
                    [st.session_state.master_database, new_df], ignore_index=True
                ) if not st.session_state.master_database.empty else new_df

                st.success(f"Record compiled. Assigned ID: **{new_id}**")

                if warnings:
                    for w in warnings:
                        st.warning(w)

                # Processing log
                with st.expander("📋 Processing Log", expanded=True):
                    st.dataframe(pd.DataFrame(processing_log), use_container_width=True, hide_index=True)

        # ── BATCH PROCESSING ──────────────────────────────
        elif "Batch Processing" in entry_mode:
            st.info("Upload rosters or multi-subject reports. Each subject receives a unique System_ID.")
            with st.form("batch_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=["png", "jpg", "jpeg", "pdf"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(
                    f"Process Batch via {selected_model.split(' ')[0]}", type="primary"
                )

            if submitted and uploaded_files:
                final_prompt = build_final_prompt(
                    st.session_state.user_prompt,
                    st.session_state.abbreviations,
                    st.session_state.extra_rules,
                    st.session_state.anti_rules
                )

                with st.spinner("Pre-processing documents…"):
                    ready_images: list[tuple[str, bytes]] = []
                    for f in uploaded_files:
                        if f.name.lower().endswith(".pdf"):
                            pages = convert_pdf_to_images(f.getvalue())
                            ready_images.extend((f"{f.name} p{i+1}", img) for i, img in enumerate(pages))
                        else:
                            ready_images.append((f.name, f.getvalue()))

                all_records: list[dict] = []
                processing_log: list[dict] = []
                progress = st.progress(0, text="Extracting batch…")

                roster_suffix = " CRITICAL: Extract EVERY subject as a separate JSON object in the array."
                for idx, (label, img_bytes) in enumerate(ready_images):
                    progress.progress(idx / len(ready_images), text=f"Analysing: {label}")
                    raw_json, api_error = blueprint_decoder(
                        img_bytes,
                        st.session_state.schema_input,
                        final_prompt + roster_suffix,
                        selected_model
                    )
                    time.sleep(4.5)

                    if api_error:
                        processing_log.append({"file": label, "status": "❌ Failed", "records": 0, "detail": api_error})
                        continue

                    records, parse_error = parse_ai_json_safe(raw_json)
                    if parse_error or not records:
                        processing_log.append({"file": label, "status": "⚠️ No data", "records": 0, "detail": parse_error or "Empty"})
                        continue

                    valid_records = []
                    for rec in records:
                        normed = normalize_record(rec, expected_cols)
                        if any(is_valid_value(v) for v in normed.values()):
                            valid_records.append(normed)

                    all_records.extend(valid_records)
                    processing_log.append({"file": label, "status": "✅ OK", "records": len(valid_records), "detail": ""})

                progress.progress(1.0, text="Done.")

                if all_records:
                    batch_df = pd.DataFrame(all_records)
                    batch_df, warnings = validate_and_clean_dataframe(batch_df, expected_cols)
                    batch_df, dupes_removed = remove_duplicates(batch_df)
                    batch_df = add_audit_columns(batch_df, username)

                    existing_ids = set(
                        st.session_state.master_database["System_ID"].astype(str).tolist()
                    ) if not st.session_state.master_database.empty and "System_ID" in st.session_state.master_database.columns else set()

                    new_ids = []
                    for _ in range(len(batch_df)):
                        nid = generate_unique_id(existing_ids)
                        new_ids.append(nid)
                        existing_ids.add(nid)

                    batch_df.insert(0, "System_ID", new_ids)

                    st.session_state.master_database = pd.concat(
                        [st.session_state.master_database, batch_df], ignore_index=True
                    ) if not st.session_state.master_database.empty else batch_df

                    st.success(f"Batch complete. Extracted **{len(batch_df)}** records.")
                    if dupes_removed:
                        st.info(f"{dupes_removed} duplicate row(s) removed automatically.")
                    for w in (warnings or []):
                        st.warning(w)
                else:
                    st.warning("No valid records were extracted from the uploaded documents.")

                with st.expander("📋 Processing Log", expanded=True):
                    st.dataframe(pd.DataFrame(processing_log), use_container_width=True, hide_index=True)

        # ── UPDATE EXISTING ───────────────────────────────
        elif "Update Existing" in entry_mode:
            st.info("Append new documentation to an existing System_ID.")
            with st.form("update_form", clear_on_submit=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_id = st.text_input("System_ID Reference", placeholder="CR-XXXX")
                with col2:
                    update_files = st.file_uploader(
                        "Upload Appendices",
                        type=["png", "jpg", "jpeg", "pdf"],
                        accept_multiple_files=True
                    )
                update_submitted = st.form_submit_button(
                    f"Update via {selected_model.split(' ')[0]}", type="primary"
                )

            if update_submitted:
                if not target_id:
                    st.error("System_ID is required.")
                elif (
                    st.session_state.master_database.empty
                    or target_id not in st.session_state.master_database.get("System_ID", pd.Series()).values
                ):
                    st.error(f"System_ID '{target_id}' not found locally. Pull from cloud first.")
                elif not update_files:
                    st.error("No documents uploaded.")
                else:
                    final_prompt = build_final_prompt(
                        st.session_state.user_prompt,
                        st.session_state.abbreviations,
                        st.session_state.extra_rules,
                        st.session_state.anti_rules
                    )

                    with st.spinner("Pre-processing…"):
                        ready_images: list[tuple[str, bytes]] = []
                        for f in update_files:
                            if f.name.lower().endswith(".pdf"):
                                pages = convert_pdf_to_images(f.getvalue())
                                ready_images.extend((f"{f.name} p{i+1}", img) for i, img in enumerate(pages))
                            else:
                                ready_images.append((f.name, f.getvalue()))

                    processing_log: list[dict] = []
                    row_idx = st.session_state.master_database.index[
                        st.session_state.master_database["System_ID"] == target_id
                    ].tolist()[0]

                    progress = st.progress(0, text="Updating record…")
                    for i, (label, img_bytes) in enumerate(ready_images):
                        progress.progress(i / len(ready_images), text=f"Processing: {label}")
                        raw_json, api_error = blueprint_decoder(
                            img_bytes, st.session_state.schema_input, final_prompt, selected_model
                        )
                        time.sleep(4.5)

                        if api_error:
                            processing_log.append({"file": label, "status": "❌ Failed", "records": 0, "detail": api_error})
                            continue

                        records, parse_error = parse_ai_json_safe(raw_json)
                        if parse_error or not records:
                            processing_log.append({"file": label, "status": "⚠️ No data", "records": 0, "detail": parse_error or "Empty"})
                            continue

                        update_rec = records[0] if records else {}
                        updated_fields = 0
                        for col in expected_cols:
                            new_val = str(update_rec.get(col, "N/A")).strip()
                            if is_valid_value(new_val):
                                st.session_state.master_database.at[row_idx, col] = new_val
                                updated_fields += 1

                        processing_log.append({"file": label, "status": "✅ OK", "records": updated_fields, "detail": f"{updated_fields} field(s) updated"})

                    progress.progress(1.0, text="Done.")
                    st.success(f"Record **{target_id}** updated.")

                    with st.expander("📋 Processing Log", expanded=True):
                        st.dataframe(pd.DataFrame(processing_log), use_container_width=True, hide_index=True)

        # ── DATA VERIFICATION TABLE ───────────────────────
        if not st.session_state.master_database.empty:
            st.divider()
            st.subheader("Data Verification Table")
            st.session_state.master_database = st.data_editor(
                st.session_state.master_database,
                num_rows="dynamic",
                use_container_width=True,
                key="data_verifier"
            )

        # ── CLOUD SYNC CONTROLS ───────────────────────────
        st.divider()
        st.subheader("Cloud Synchronisation")
        col_commit, col_pull_btn, col_export = st.columns(3)

        with col_commit:
            if st.button("☁️ Commit to Cloud", type="primary", use_container_width=True):
                if st.session_state.master_database.empty:
                    st.warning("Local cache is empty.")
                elif not active_sheet_url:
                    st.error("No Google Sheet URL configured.")
                else:
                    # Final dedup pass before commit
                    st.session_state.master_database, dupes = remove_duplicates(st.session_state.master_database)
                    if dupes:
                        st.info(f"Removed {dupes} duplicate(s) before commit.")

                    # Ensure all expected columns exist
                    final_cols = ["System_ID"] + [c for c in expected_cols if c.lower() != "system_id"]
                    for col in final_cols:
                        if col not in st.session_state.master_database.columns:
                            st.session_state.master_database[col] = "N/A"
                    st.session_state.master_database = st.session_state.master_database[
                        [c for c in final_cols if c in st.session_state.master_database.columns]
                    ]

                    missing_ids = st.session_state.master_database["System_ID"].astype(str).str.strip().str.lower().isin(INVALID_VALUES)
                    if missing_ids.any():
                        st.error(f"{missing_ids.sum()} record(s) are missing a System_ID. Resolve before committing.")
                    else:
                        with st.spinner("Merging and committing to Google Sheets…"):
                            try:
                                push_to_sheet_merge(
                                    st.session_state.master_database, active_sheet_url, project_tab
                                )
                                st.toast("Cloud commit successful.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Commit failed: {exc}")
                                logger.error("Commit error: %s", exc)

        with col_pull_btn:
            if st.button("⬇️ Pull from Cloud", use_container_width=True):
                if not active_sheet_url:
                    st.error("No Google Sheet URL configured.")
                else:
                    with st.spinner("Downloading from cloud…"):
                        try:
                            st.session_state.master_database = pull_from_sheet(active_sheet_url, project_tab)
                            st.toast("Cache synchronised with cloud.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Pull failed: {exc}")

        with col_export:
            if not st.session_state.master_database.empty:
                csv_bytes = st.session_state.master_database.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export CSV",
                    data=csv_bytes,
                    file_name=f"{project_tab}_export_{datetime.utcnow().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # ═══════════════════════════════════════════════════════
    # TAB 2 — DATA EXPLORER
    # ═══════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Data Explorer")

        if st.session_state.master_database.empty:
            st.info("Synchronise with cloud or process records to enable analytics.")
        else:
            df = st.session_state.master_database.copy()

            # Standardise placeholders and attempt numeric coercion
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(
                        {"nan": "N/A", "None": "N/A", "NaN": "N/A", "none": "N/A", "null": "N/A", "": "N/A"},
                        regex=False
                    )
                df[col] = pd.to_numeric(df[col].replace("N/A", pd.NA), errors="ignore")

            all_cols = [c for c in df.columns if c != "System_ID"]

            # ── SEARCH & FILTER ───────────────────────────
            st.markdown("#### 🔍 Search & Filter")
            search_col1, search_col2 = st.columns([2, 1])
            with search_col1:
                global_search = st.text_input("Global search (searches all columns)", placeholder="Type to filter rows…")
            with search_col2:
                filter_col = st.selectbox("Filter by column", ["— None —"] + all_cols)

            filter_df = df.copy()

            if global_search.strip():
                mask = filter_df.astype(str).apply(
                    lambda col: col.str.contains(global_search.strip(), case=False, na=False)
                ).any(axis=1)
                filter_df = filter_df[mask]

            if filter_col != "— None —":
                unique_vals = sorted(
                    filter_df[filter_col].dropna().astype(str).unique().tolist()
                )
                selected_vals = st.multiselect(f"Values for '{filter_col}'", unique_vals, default=unique_vals[:10] if len(unique_vals) > 10 else unique_vals)
                if selected_vals:
                    filter_df = filter_df[filter_df[filter_col].astype(str).isin(selected_vals)]

            st.caption(f"Showing **{len(filter_df)}** of **{len(df)}** records.")
            st.dataframe(filter_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── GROUPING ──────────────────────────────────
            st.markdown("#### 📊 Grouping & Counts")
            group_col = st.selectbox("Group by", ["— None —"] + all_cols, key="group_sel")
            if group_col != "— None —":
                group_counts = (
                    filter_df[group_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                group_counts.columns = [group_col, "Count"]
                group_counts = group_counts[group_counts[group_col] != "N/A"]
                st.dataframe(group_counts, use_container_width=True, hide_index=True)

            st.divider()

            # ── VISUALISATION ─────────────────────────────
            st.markdown("#### 📈 Visualisation")
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                x_axis = st.selectbox("Categorical Axis (X)", all_cols, key="x_sel")
            with viz_col2:
                y_axis = st.selectbox("Numerical Axis (Y)", all_cols, key="y_sel")

            chart_type = st.radio("Chart Type", ["Bar", "Pie", "Scatter"], horizontal=True)

            try:
                chart_df = filter_df[
                    ~filter_df[x_axis].astype(str).isin(["N/A", "nan"])
                    & ~filter_df[y_axis].astype(str).isin(["N/A", "nan"])
                ].copy()

                if chart_df.empty:
                    st.warning("No plottable data with current filters.")
                else:
                    if chart_type == "Bar":
                        fig = px.bar(chart_df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}")
                    elif chart_type == "Pie":
                        fig = px.pie(chart_df, names=x_axis, title=f"Distribution of {x_axis}")
                    else:
                        fig = px.scatter(chart_df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} vs {x_axis}")

                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Visualisation failed: {exc}")
                logger.warning("Chart error: %s", exc)
