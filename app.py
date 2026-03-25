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
    'hb', 'plt', 'mic', 'los', 'temperature', 'bp', 'hr', 'rr', 'spo2',
    'hba1c', 'glucose', 'cholesterol', 'triglycerides', 'ldl', 'hdl',
    'creatinine', 'egfr', 'sodium', 'potassium', 'albumin', 'bilirubin'
]

DEFAULT_PROMPT = (
    "You are an expert clinical data extraction engine. "
    "Extract all structured medical and demographic data visible in this document image. "
    "Apply standard clinical abbreviation expansion. "
    "Return ONLY a valid JSON array — no explanation, no preamble, no markdown fences."
)

DEFAULT_BUILT_IN_RULES = (
    "Expand common abbreviations (M=Male, F=Female, HTN=Hypertension, DM=Diabetes Mellitus, "
    "HbA1c=Glycated Haemoglobin, FBS=Fasting Blood Sugar, RBS=Random Blood Sugar). "
    "Standardize units where visible. "
    "Use exactly the string 'N/A' for any field that is missing or unreadable. "
    "Never fabricate or infer data not explicitly present in the document."
)

# ── Per-model image compression settings ─────────────────────
# OpenAI paid tier: highest fidelity — larger canvas, near-lossless
# Gemini / Groq:    balanced — good quality, lower token cost
COMPRESSION_PROFILES = {
    "openai": {
        "max_size": 1920,   # Full HD ceiling — preserves table detail
        "quality":  95,     # Near-lossless JPEG
    },
    "gemini": {
        "max_size": 1280,
        "quality":  88,
    },
    "groq": {
        "max_size": 1024,
        "quality":  85,
    },
}

PROMPT_TEMPLATES = {
    "— Select a Template —": {
        "schema": "", "prompt": "", "abbreviations": "", "rules": "", "anti": ""
    },
    "🦠 Microbiology": {
        "schema": "Patient_ID, Age, Gender, Organism, Antibiotic, MIC, Resistance_Pattern, Specimen_Type, Culture_Date, Outcome",
        "prompt": "Extract microbiological culture and sensitivity data. Focus on organism identity, antibiotic susceptibility, and clinical outcome.",
        "abbreviations": "MIC=Minimum Inhibitory Concentration, MDR=Multi-Drug Resistant, MRSA=Methicillin-Resistant S. aureus, S=Sensitive, R=Resistant, I=Intermediate, ESBL=Extended-Spectrum Beta-Lactamase",
        "rules": "Record each organism–antibiotic pair as a separate entry. If multiple organisms are present, create separate entries for each.",
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
        "abbreviations": (
            "WBC=White Blood Cells, Hb=Hemoglobin, PLT=Platelets, CRP=C-Reactive Protein, "
            "ESR=Erythrocyte Sedimentation Rate, HbA1c=Glycated Hemoglobin, eGFR=Estimated Glomerular Filtration Rate, "
            "FBS=Fasting Blood Sugar, RBS=Random Blood Sugar, LDL=Low-Density Lipoprotein, HDL=High-Density Lipoprotein"
        ),
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
# 2. UI SETUP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="CloudResearch Command Center", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.markdown("""
    <style>
        h1  { font-size: 1.5rem  !important; font-weight: 700 !important; padding-bottom: 0.5rem !important; }
        h2  { font-size: 1.1rem  !important; font-weight: 600 !important; padding-top: 1rem !important; padding-bottom: 0.2rem !important; }
        h3  { font-size: 1.05rem !important; font-weight: 600 !important; padding-bottom: 0.2rem !important; }
        .streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.95rem !important; }
        .stAlert { border-radius: 6px !important; }
        .debug-box {
            background: #1a1a2e; color: #00ff88; padding: 12px;
            border-radius: 6px; font-family: monospace; font-size: 0.8rem;
            border-left: 3px solid #00ff88; margin: 4px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 3. HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════

def is_valid_value(value):
    return str(value).strip().lower() not in INVALID_VALUES


def generate_unique_id(existing_ids):
    alphabet = string.ascii_uppercase + string.digits
    while True:
        candidate = "CR-" + "".join(random.choices(alphabet, k=4))
        if candidate not in existing_ids:
            return candidate


def debug_log(label, content, debug_mode):
    if debug_mode:
        st.markdown(
            f'<div class="debug-box"><b>🔍 {label}</b><br>'
            f'<pre>{str(content)[:1500]}</pre></div>',
            unsafe_allow_html=True
        )


def parse_ai_json_safe(raw_text):
    """
    Robustly parse AI output into a list of dicts.
    Returns (records_list, error_message_or_None).

    Handles:
    - Markdown code fences  ``` json ... ```
    - Trailing prose after the JSON block
    - Wrapped objects  {"data": [...]}
    - Single dict responses
    - Empty or completely non-JSON responses
    - Trailing commas (common AI mistake)
    """
    if not raw_text or not raw_text.strip():
        return [], "AI returned an empty response."

    cleaned = raw_text.strip()

    # Strip markdown code fences
    if "```" in cleaned:
        parts      = re.split(r"```(?:json)?", cleaned)
        candidates = [p.strip() for p in parts if p.strip().startswith(("[", "{"))]
        cleaned    = candidates[0] if candidates else cleaned

    # Extract the FIRST complete balanced JSON structure.
    # Character-level scan stops at the matching bracket so trailing
    # prose (e.g. "Let me know if you need help!") never corrupts the parse.
    json_str = None
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        depth       = 0
        in_string   = False
        escape_next = False
        for i, ch in enumerate(cleaned[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        json_str = cleaned[start:i + 1]
                        break
        if json_str:
            break

    if not json_str:
        return [], f"No valid JSON found. Raw snippet: {cleaned[:200]}"

    # Primary parse attempt; fallback strips trailing commas
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            logger.warning("JSON decode failed: %s | raw: %s", exc, json_str[:300])
            return [], f"JSON parse error: {exc}"

    # Unwrap common envelope patterns
    if isinstance(parsed, dict):
        for key in ("data", "records", "patients", "results", "entries", "output"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key], None
        return [parsed], None

    if isinstance(parsed, list):
        return parsed, None

    return [], f"Unexpected JSON root type: {type(parsed).__name__}"


def normalize_record(record, expected_cols):
    """
    Map AI-returned keys to schema columns.
    Handles: case differences, trailing colons, spaces, underscores vs spaces.
    """
    def normalise_key(k):
        return re.sub(r"[\s_\-]+", "_", str(k).lower().strip().rstrip(":"))

    lower_map  = {normalise_key(k): v for k, v in record.items()}
    normalised = {}
    for col in expected_cols:
        col_key  = normalise_key(col)
        value    = lower_map.get(col_key, "N/A")
        str_val  = str(value).strip()
        normalised[col] = str_val if is_valid_value(str_val) else "N/A"
    return normalised


def validate_and_clean_dataframe(df, expected_cols):
    warnings_list = []
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "N/A"
            warnings_list.append(f"Column '{col}' was missing — added with default values.")

    for col in df.columns:
        if any(kw in col.lower() for kw in NUMERIC_KEYWORDS):
            original = df[col].copy()
            coerced  = pd.to_numeric(df[col].replace("N/A", pd.NA), errors="coerce")
            failed   = coerced.isna() & original.notna() & (original != "N/A")
            if failed.any():
                sample = original[failed].iloc[0]
                warnings_list.append(
                    f"Column '{col}': {failed.sum()} non-numeric value(s) kept as-is "
                    f"(e.g. '{sample}')."
                )

    placeholder_map = {v: "N/A" for v in {
        "nan", "None", "NaN", "none", "null", "NULL", "NA", "N/A", ""
    }}
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace(placeholder_map, regex=False)

    return df, warnings_list


def add_audit_columns(df, username):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if "Created_By" not in df.columns:
        df["Created_By"] = username
    if "Timestamp" not in df.columns:
        df["Timestamp"] = ts
    return df


def remove_duplicates(df):
    if df.empty:
        return df, 0
    key_cols = [c for c in df.columns if c not in {"System_ID", "Timestamp", "Created_By"}]
    before   = len(df)
    df       = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return df, before - len(df)

# ═══════════════════════════════════════════════════════════════
# 4. GOOGLE SHEETS INTEGRATION
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def get_google_sheet_client():
    scope      = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds      = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)


def _get_or_create_worksheet(sheet_url, tab_name):
    client = get_google_sheet_client()
    try:
        return client.open_by_url(sheet_url).worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = client.open_by_url(sheet_url)
        return spreadsheet.add_worksheet(title=tab_name, rows="1000", cols="30")


def pull_from_sheet(sheet_url, tab_name):
    sheet      = _get_or_create_worksheet(sheet_url, tab_name)
    cloud_data = sheet.get_all_values()

    if len(cloud_data) > 1:
        cloud_df = pd.DataFrame(cloud_data[1:], columns=cloud_data[0])
    elif len(cloud_data) == 1:
        cloud_df = pd.DataFrame(columns=cloud_data[0])
    else:
        return pd.DataFrame()

    if cloud_df.empty:
        return cloud_df

    cloud_df.rename(
        columns=lambda x: "System_ID"
        if str(x).strip().lower() in {"system_id", "system id"} else x,
        inplace=True
    )

    if "System_ID" not in cloud_df.columns:
        existing = set()
        new_ids  = []
        for _ in range(len(cloud_df)):
            nid = generate_unique_id(existing)
            new_ids.append(nid)
            existing.add(nid)
        cloud_df.insert(0, "System_ID", new_ids)
    else:
        missing_mask = (
            cloud_df["System_ID"].astype(str).str.strip().str.lower().isin(INVALID_VALUES)
        )
        if missing_mask.any():
            existing = set(cloud_df.loc[~missing_mask, "System_ID"].astype(str).tolist())
            new_ids  = []
            for _ in range(missing_mask.sum()):
                nid = generate_unique_id(existing)
                new_ids.append(nid)
                existing.add(nid)
            cloud_df.loc[missing_mask, "System_ID"] = new_ids

    return cloud_df


def push_to_sheet_merge(local_df, sheet_url, tab_name):
    """
    Safe merge-push:
    1. Pull existing cloud data
    2. Keep cloud rows NOT in local_df (cloud-only records survive)
    3. Concat remainder + local, then overwrite
    """
    sheet = _get_or_create_worksheet(sheet_url, tab_name)
    try:
        cloud_df = pull_from_sheet(sheet_url, tab_name)
    except Exception as exc:
        logger.warning("Could not pull before push: %s", exc)
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
    cols      = ["System_ID"] + [c for c in merged_df.columns if c != "System_ID"]
    merged_df = merged_df[[c for c in cols if c in merged_df.columns]]

    sheet.clear()
    if not merged_df.empty:
        sheet.update(
            range_name="A1",
            values=[merged_df.columns.tolist()] + merged_df.values.tolist()
        )
    else:
        sheet.update(range_name="A1", values=[["System_ID"]])

    return merged_df

# ═══════════════════════════════════════════════════════════════
# 5. PROFILE SYNC
# ═══════════════════════════════════════════════════════════════

_PROFILE_COLUMNS = [
    "Username", "Target_Sheet_URL", "Schema", "Prompt",
    "Abbreviations", "Extra_Rules", "Anti_Rules"
]


def sync_user_profile(username, mode="pull", profile_data=None):
    admin_url = st.secrets.get("ADMIN_SHEET_URL", "")
    if not admin_url:
        return None

    client = get_google_sheet_client()
    try:
        sheet = client.open_by_url(admin_url).worksheet("Profiles")
    except gspread.exceptions.WorksheetNotFound:
        spreadsheet = client.open_by_url(admin_url)
        sheet       = spreadsheet.add_worksheet(title="Profiles", rows="100", cols="10")
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

def compress_image(image_bytes, model_key="gemini"):
    """
    Model-aware compression using COMPRESSION_PROFILES.

    OpenAI paid:  1920px / quality 95  — maximum fidelity for dense lab reports
    Gemini:       1280px / quality 88  — balanced quality vs token cost
    Groq:         1024px / quality 85  — conservative for free tier limits

    Always converts to RGB so JPEG encoding never errors on RGBA/P/CMYK inputs.
    Uses LANCZOS resampling (highest quality downsample filter in Pillow).
    """
    profile = COMPRESSION_PROFILES.get(model_key, COMPRESSION_PROFILES["gemini"])
    img     = Image.open(io.BytesIO(image_bytes))

    # Normalise colour space — JPEG cannot encode RGBA, P, or CMYK
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Downsample only if needed — never upscale
    if img.width > profile["max_size"] or img.height > profile["max_size"]:
        img = img.copy()
        img.thumbnail((profile["max_size"], profile["max_size"]), Image.LANCZOS)

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=profile["quality"], optimize=True)
    return output.getvalue()


def convert_pdf_to_images(pdf_bytes):
    images = []
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix  = page.get_pixmap(dpi=200)   # Raised from 150 → 200 dpi for dense lab reports
        images.append(pix.tobytes("jpeg"))
    return images

# ═══════════════════════════════════════════════════════════════
# 7. PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_final_prompt(user_prompt, abbreviations, extra_rules, anti_rules):
    effective_prompt = user_prompt.strip() or DEFAULT_PROMPT
    effective_rules  = extra_rules.strip()  or DEFAULT_BUILT_IN_RULES
    abbrev_block     = f"Abbreviation map: {abbreviations.strip()}." if abbreviations.strip() else ""
    anti_block       = f"Do NOT extract: {anti_rules.strip()}."       if anti_rules.strip()  else ""

    return (
        f"TASK: Extract clinical data and return a JSON array.\n"
        f"DIRECTIVE: {effective_prompt}\n"
        f"{abbrev_block}\n"
        f"RULES: {effective_rules}\n"
        f"{anti_block}\n"
        f"OUTPUT FORMAT: A valid JSON array only, like [{{\"key\": \"value\"}}]. "
        f"One JSON object per patient/subject. "
        f"Use 'N/A' for any missing field. "
        f"Return ONLY the JSON array. No explanation. No markdown. No prose before or after."
    ).strip()

# ═══════════════════════════════════════════════════════════════
# 8. AI EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════════

def _model_key_from_choice(model_choice):
    """Map display name to compression profile key."""
    mc = model_choice.lower()
    if "gemini" in mc:
        return "gemini"
    if "openai" in mc:
        return "openai"
    return "groq"


def blueprint_decoder(image_bytes, schema_columns, final_prompt, model_choice, debug_mode=False):
    """
    Send image to selected AI model.  Returns (raw_json_string, error_or_None).

    Key design decisions
    ────────────────────
    • Rate-limit sleep fires BEFORE the API call (not after) to protect the
      first request in a batch from hitting a 429.
    • OpenAI:  uses gpt-4o (not mini) and does NOT use response_format
      json_object.  That mode requires a dict root and the wrapping instruction
      was confusing the model on field-level extraction (e.g. HbA1c missing).
      Instead the prompt enforces raw array output and parse_ai_json_safe handles it.
    • Gemini:  no response_mime_type — it forced object wrapping that broke
      array parsing.
    • All models receive the same prompt terminator:
      "Return ONLY the JSON array. No explanation. No markdown."
    """
    full_prompt = (
        f"{final_prompt}\n\n"
        f"REQUIRED JSON KEYS (use EXACTLY these as your key names, no changes): "
        f"{schema_columns}\n\n"
        f"STRICT OUTPUT RULE: Your entire response must be a single raw JSON array.\n"
        f"Start your response with [ and end with ].\n"
        f"Do NOT use markdown code blocks.\n"
        f"Do NOT write anything before or after the array.\n"
        f"If the image is unreadable, return: []"
    )

    # Throttle BEFORE the call — protects first request in a multi-image batch
    time.sleep(4.5)

    model_key  = _model_key_from_choice(model_choice)
    compressed = compress_image(image_bytes, model_key=model_key)
    b64_image  = base64.b64encode(compressed).decode("utf-8")

    debug_log("Prompt Sent", full_prompt, debug_mode)
    debug_log(
        "Image Stats",
        f"Size after compression: {len(compressed)/1024:.1f} KB  |  "
        f"Profile: {model_key} — {COMPRESSION_PROFILES[model_key]}",
        debug_mode
    )

    try:
        # ── GEMINI ───────────────────────────────────────────
        if "Gemini" in model_choice:
            img               = Image.open(io.BytesIO(compressed))
            gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
            model             = genai.GenerativeModel(gemini_model_name)
            # No response_mime_type — it wraps output in a dict root which
            # conflicts with array-first parsing and silently returns nothing.
            response          = model.generate_content([full_prompt, img])
            raw               = response.text.strip()
            debug_log("Gemini Raw Response", raw, debug_mode)
            return raw, None

        # ── OPENAI ───────────────────────────────────────────
        elif "OpenAI" in model_choice:
            client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
            # FIX 1: gpt-4o instead of gpt-4o-mini
            #   gpt-4o-mini has significantly weaker vision; it routinely misses
            #   small-font lab values like HbA1c on dense result tables.
            # FIX 2: No response_format=json_object
            #   That mode mandates a dict root, so we had to add a "wrap in data key"
            #   instruction which was adding cognitive load and causing the model to
            #   focus on structure over content — exactly why HbA1c was being skipped.
            #   Without it, gpt-4o follows the prompt and returns a raw array directly,
            #   which parse_ai_json_safe handles perfectly.
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": full_prompt},
                        {"type": "image_url", "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high"   # Use high-detail vision for dense lab tables
                        }}
                    ]
                }]
            )
            raw = response.choices[0].message.content.strip()
            debug_log("OpenAI Raw Response", raw, debug_mode)
            return raw, None

        # ── GROQ ─────────────────────────────────────────────
        elif "Groq" in model_choice:
            client   = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": full_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }}
                    ]
                }]
            )
            raw = response.choices[0].message.content.strip()
            debug_log("Groq Raw Response", raw, debug_mode)
            return raw, None

        return "[]", f"Unknown model: {model_choice}"

    except Exception as exc:
        error_msg = f"{model_choice} API error: {exc}"
        logger.error(error_msg)
        debug_log("API Exception", error_msg, debug_mode)
        return "[]", error_msg

# ═══════════════════════════════════════════════════════════════
# 9. SHARED PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════

def prepare_images_from_uploads(uploaded_files):
    ready = []
    for f in uploaded_files:
        if f.name.lower().endswith(".pdf"):
            pages = convert_pdf_to_images(f.getvalue())
            ready.extend([(f"{f.name} — page {i+1}", img) for i, img in enumerate(pages)])
        else:
            ready.append((f.name, f.getvalue()))
    return ready


def run_extraction_pipeline(
    ready_images, schema_columns, expected_cols,
    final_prompt, model_choice, mode="batch", debug_mode=False
):
    """
    Core extraction loop.
    mode='single' → merge all pages into one master record dict
    mode='batch'  → each page may produce multiple separate records
    Returns (result, processing_log)
    """
    processing_log = []
    progress_bar   = st.progress(0, text="Starting extraction…")

    # ── BATCH ────────────────────────────────────────────────
    if mode == "batch":
        all_records = []
        for idx, (label, img_bytes) in enumerate(ready_images):
            progress_bar.progress(idx / len(ready_images), text=f"Analysing: {label}")
            raw_json, api_error = blueprint_decoder(
                img_bytes, schema_columns, final_prompt, model_choice, debug_mode
            )
            if api_error:
                processing_log.append({
                    "File": label, "Status": "❌ API Error", "Records": 0, "Detail": api_error
                })
                continue
            records, parse_error = parse_ai_json_safe(raw_json)
            debug_log(f"Parsed [{label}]", json.dumps(records[:2], indent=2), debug_mode)
            if parse_error or not records:
                processing_log.append({
                    "File": label, "Status": "⚠️ No Data",
                    "Records": 0, "Detail": parse_error or "AI returned empty output"
                })
                continue
            valid = [
                normalize_record(rec, expected_cols)
                for rec in records
                if any(is_valid_value(v) for v in normalize_record(rec, expected_cols).values())
            ]
            all_records.extend(valid)
            processing_log.append({
                "File": label, "Status": "✅ OK",
                "Records": len(valid), "Detail": f"{len(valid)} subject(s) extracted"
            })
        progress_bar.progress(1.0, text="Extraction complete.")
        if not all_records:
            return pd.DataFrame(), processing_log
        return pd.DataFrame(all_records), processing_log

    # ── SINGLE ───────────────────────────────────────────────
    elif mode == "single":
        master = {col: "N/A" for col in expected_cols}
        for idx, (label, img_bytes) in enumerate(ready_images):
            progress_bar.progress(idx / len(ready_images), text=f"Processing: {label}")
            raw_json, api_error = blueprint_decoder(
                img_bytes, schema_columns, final_prompt, model_choice, debug_mode
            )
            if api_error:
                processing_log.append({
                    "File": label, "Status": "❌ API Error", "Records": 0, "Detail": api_error
                })
                continue
            records, parse_error = parse_ai_json_safe(raw_json)
            debug_log(f"Parsed [{label}]", json.dumps(records[:2], indent=2), debug_mode)
            if parse_error or not records:
                processing_log.append({
                    "File": label, "Status": "⚠️ No Data",
                    "Records": 0, "Detail": parse_error or "AI returned empty output"
                })
                continue
            updated_fields = 0
            for rec in records:
                normed = normalize_record(rec, expected_cols)
                for col in expected_cols:
                    new_val = normed.get(col, "N/A")
                    if is_valid_value(new_val) and not is_valid_value(master[col]):
                        pass  # keep existing non-N/A value
                    elif is_valid_value(new_val):
                        master[col]     = new_val
                        updated_fields += 1
            processing_log.append({
                "File": label, "Status": "✅ OK",
                "Records": 1, "Detail": f"{updated_fields} field(s) filled"
            })
        progress_bar.progress(1.0, text="Compilation complete.")
        return master, processing_log

# ═══════════════════════════════════════════════════════════════
# 10. AUTHENTICATION
# ═══════════════════════════════════════════════════════════════

def _deep_copy_dict(obj):
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
# 11. MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

auth_status = st.session_state.get("authentication_status")

if auth_status is False:
    st.error("Incorrect username or password.")

elif auth_status is None:
    st.title("CloudResearch")
    st.warning("Enter your credentials to access the Command Center.")

elif auth_status is True:
    username = st.session_state.get("username", "")
    name     = st.session_state.get("name", "")

    # ── SESSION INIT ─────────────────────────────────────────
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

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.success(f"✅ {name}")
        authenticator.logout("Logout", "sidebar")
        st.divider()

        with st.expander("⚙️ Setup", expanded=True):
            selected_model = st.selectbox(
                "AI Model",
                ["Google Gemini (Primary)", "Groq (Free Fallback)", "OpenAI (Paid Fallback)"]
            )

            # Show active compression profile so users know what's being used
            active_profile_key = _model_key_from_choice(selected_model)
            active_profile     = COMPRESSION_PROFILES[active_profile_key]
            st.caption(
                f"🖼️ Image profile: **{active_profile['max_size']}px** / "
                f"**{active_profile['quality']}% quality**"
            )

            active_sheet_url = st.text_input(
                "Google Sheet URL", value=st.session_state.target_sheet_url
            )
            st.session_state.target_sheet_url = active_sheet_url

            debug_mode = st.toggle(
                "🐛 Debug Mode", value=False,
                help=(
                    "Shows the exact prompt sent, compressed image size, "
                    "raw AI response, and parse results for every file."
                )
            )

        with st.expander("📋 Schema", expanded=True):
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
                if st.button("↓ Pull", use_container_width=True):
                    if active_sheet_url:
                        with st.spinner("Pulling schema…"):
                            try:
                                sheet   = get_google_sheet_client() \
                                    .open_by_url(active_sheet_url).worksheet(username)
                                headers = sheet.row_values(1)
                                clean   = [c for c in headers if c.lower() != "system_id"]
                                st.session_state.safe_schema_val = ", ".join(clean)
                                st.session_state.schema_input    = st.session_state.safe_schema_val
                                st.toast("Schema pulled.")
                                st.rerun()
                            except Exception as exc:
                                st.toast(f"Pull failed: {exc}")
            with col_push:
                if st.button("↑ Push", use_container_width=True):
                    if active_sheet_url:
                        with st.spinner("Pushing schema…"):
                            try:
                                sheet   = get_google_sheet_client() \
                                    .open_by_url(active_sheet_url).worksheet(username)
                                cols    = [
                                    c.strip()
                                    for c in st.session_state.safe_schema_val.split(",")
                                    if c.strip()
                                ]
                                headers = ["System_ID"] + [
                                    c for c in cols if c.lower() != "system_id"
                                ]
                                sheet.update(range_name="A1", values=[headers])
                                st.toast("Schema pushed.")
                            except Exception as exc:
                                st.toast(f"Push failed: {exc}")

            updated_schema = st.text_input(
                "Columns (comma-separated)", value=st.session_state.safe_schema_val
            )
            st.session_state.safe_schema_val = updated_schema
            st.session_state.schema_input    = updated_schema

        with st.expander("🧠 Extraction Logic"):
            st.session_state.user_prompt = st.text_area(
                "Primary Directive", value=st.session_state.user_prompt, height=120
            )
            st.session_state.abbreviations = st.text_area(
                "Abbreviations Map", value=st.session_state.abbreviations, height=80
            )
            st.session_state.extra_rules = st.text_area(
                "Inclusion Rules", value=st.session_state.extra_rules, height=80
            )
            st.session_state.anti_rules = st.text_area(
                "Exclusion Rules", value=st.session_state.anti_rules, height=60
            )

        with st.expander("💾 Profile"):
            if st.button("Save Configuration", type="primary", use_container_width=True):
                with st.spinner("Saving…"):
                    sync_user_profile(username, mode="push", profile_data={
                        "Target_Sheet_URL": st.session_state.target_sheet_url,
                        "Schema":           st.session_state.schema_input,
                        "Prompt":           st.session_state.user_prompt,
                        "Abbreviations":    st.session_state.abbreviations,
                        "Extra_Rules":      st.session_state.extra_rules,
                        "Anti_Rules":       st.session_state.anti_rules,
                    })
                    st.success("Profile saved.")

        with st.expander("🔧 System"):
            st.warning("Clears all unsaved local data.")
            if st.button("Purge Local Cache", use_container_width=True):
                st.session_state.master_database = pd.DataFrame()
                st.rerun()

    # ── MAIN AREA ────────────────────────────────────────────
    st.title("CloudResearch Command Center")

    if debug_mode:
        st.info(
            "🐛 **Debug Mode ON** — the exact prompt, image stats, raw AI response, "
            "and parse diagnostics appear inline below each processed file."
        )

    tabs          = st.tabs(["📥 Data Entry & Sync", "🔍 Data Explorer"])
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

        final_prompt = build_final_prompt(
            st.session_state.user_prompt,
            st.session_state.abbreviations,
            st.session_state.extra_rules,
            st.session_state.anti_rules
        )

        if debug_mode:
            with st.expander("🔍 Preview Final Prompt"):
                st.code(final_prompt, language="text")

        # ── SINGLE RECORD ─────────────────────────────────
        if "Single Record" in entry_mode:
            st.info(
                "Upload all documents for one subject. "
                "The engine compiles all pages into one unified profile."
            )
            with st.form("single_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=["png", "jpg", "jpeg", "pdf"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(
                    f"Process via {selected_model.split()[0]}", type="primary"
                )

            if submitted and uploaded_files:
                with st.spinner("Pre-processing files…"):
                    ready_images = prepare_images_from_uploads(uploaded_files)
                st.write(f"**{len(ready_images)}** image(s) prepared.")

                master_record, proc_log = run_extraction_pipeline(
                    ready_images, st.session_state.schema_input, expected_cols,
                    final_prompt, selected_model, mode="single", debug_mode=debug_mode
                )

                if isinstance(master_record, dict) and any(
                    is_valid_value(v) for v in master_record.values()
                ):
                    new_df           = pd.DataFrame([master_record])
                    new_df, warnings = validate_and_clean_dataframe(new_df, expected_cols)
                    new_df           = add_audit_columns(new_df, username)

                    existing_ids = (
                        set(st.session_state.master_database["System_ID"].astype(str).tolist())
                        if not st.session_state.master_database.empty
                        and "System_ID" in st.session_state.master_database.columns
                        else set()
                    )
                    new_id = generate_unique_id(existing_ids)
                    new_df.insert(0, "System_ID", [new_id])

                    st.session_state.master_database = (
                        pd.concat([st.session_state.master_database, new_df], ignore_index=True)
                        if not st.session_state.master_database.empty else new_df
                    )
                    st.success(f"✅ Record compiled. Assigned ID: **{new_id}**")
                    for w in warnings:
                        st.warning(w)
                else:
                    st.error(
                        "⚠️ Extraction returned no usable data. "
                        "Enable **Debug Mode** (sidebar → ⚙️ Setup) and re-run to "
                        "inspect the raw AI response."
                    )

                with st.expander("📋 Processing Log", expanded=True):
                    st.dataframe(pd.DataFrame(proc_log), use_container_width=True, hide_index=True)

        # ── BATCH PROCESSING ──────────────────────────────
        elif "Batch Processing" in entry_mode:
            st.info(
                "Upload rosters or multi-subject reports. "
                "Each detected subject gets a unique System_ID."
            )
            with st.form("batch_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=["png", "jpg", "jpeg", "pdf"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(
                    f"Process Batch via {selected_model.split()[0]}", type="primary"
                )

            if submitted and uploaded_files:
                roster_prompt = final_prompt + (
                    "\n\nCRITICAL: This document may contain MULTIPLE subjects. "
                    "Extract EVERY subject as a SEPARATE JSON object in the array. "
                    "Do NOT merge multiple subjects into one object."
                )
                with st.spinner("Pre-processing files…"):
                    ready_images = prepare_images_from_uploads(uploaded_files)
                st.write(f"**{len(ready_images)}** image(s) prepared.")

                batch_df, proc_log = run_extraction_pipeline(
                    ready_images, st.session_state.schema_input, expected_cols,
                    roster_prompt, selected_model, mode="batch", debug_mode=debug_mode
                )

                if not batch_df.empty:
                    batch_df, warnings = validate_and_clean_dataframe(batch_df, expected_cols)
                    batch_df, n_dupes  = remove_duplicates(batch_df)
                    batch_df           = add_audit_columns(batch_df, username)

                    existing_ids = (
                        set(st.session_state.master_database["System_ID"].astype(str).tolist())
                        if not st.session_state.master_database.empty
                        and "System_ID" in st.session_state.master_database.columns
                        else set()
                    )
                    new_ids = []
                    for _ in range(len(batch_df)):
                        nid = generate_unique_id(existing_ids)
                        new_ids.append(nid)
                        existing_ids.add(nid)
                    batch_df.insert(0, "System_ID", new_ids)

                    st.session_state.master_database = (
                        pd.concat([st.session_state.master_database, batch_df], ignore_index=True)
                        if not st.session_state.master_database.empty else batch_df
                    )
                    st.success(f"✅ Batch complete. **{len(batch_df)}** record(s) extracted.")
                    if n_dupes:
                        st.info(f"🔁 {n_dupes} duplicate row(s) removed automatically.")
                    for w in warnings:
                        st.warning(w)
                else:
                    st.error(
                        "⚠️ No records extracted. Enable **Debug Mode** and re-run to diagnose."
                    )

                with st.expander("📋 Processing Log", expanded=True):
                    st.dataframe(pd.DataFrame(proc_log), use_container_width=True, hide_index=True)

        # ── UPDATE EXISTING ───────────────────────────────
        elif "Update Existing" in entry_mode:
            st.info("Append new documentation to an existing System_ID.")
            with st.form("update_form", clear_on_submit=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    target_id = st.text_input("System_ID Reference", placeholder="CR-XXXX")
                with c2:
                    update_files = st.file_uploader(
                        "Upload Appendices",
                        type=["png", "jpg", "jpeg", "pdf"],
                        accept_multiple_files=True
                    )
                update_submitted = st.form_submit_button(
                    f"Update via {selected_model.split()[0]}", type="primary"
                )

            if update_submitted:
                if not target_id:
                    st.error("System_ID is required.")
                elif (
                    st.session_state.master_database.empty
                    or target_id not in st.session_state.master_database.get(
                        "System_ID", pd.Series()
                    ).values
                ):
                    st.error(f"'{target_id}' not found locally. Pull from cloud first.")
                elif not update_files:
                    st.error("No files uploaded.")
                else:
                    with st.spinner("Pre-processing…"):
                        ready_images = prepare_images_from_uploads(update_files)

                    update_record, proc_log = run_extraction_pipeline(
                        ready_images, st.session_state.schema_input, expected_cols,
                        final_prompt, selected_model, mode="single", debug_mode=debug_mode
                    )

                    row_idx = st.session_state.master_database.index[
                        st.session_state.master_database["System_ID"] == target_id
                    ].tolist()[0]

                    updated_count = 0
                    for col in expected_cols:
                        new_val = update_record.get(col, "N/A")
                        if is_valid_value(new_val):
                            st.session_state.master_database.at[row_idx, col] = new_val
                            updated_count += 1

                    if updated_count:
                        st.success(
                            f"✅ Record **{target_id}** updated. "
                            f"{updated_count} field(s) refreshed."
                        )
                    else:
                        st.error(
                            "⚠️ Update extracted no data. "
                            "Enable Debug Mode and re-run to diagnose."
                        )

                    with st.expander("📋 Processing Log", expanded=True):
                        st.dataframe(
                            pd.DataFrame(proc_log), use_container_width=True, hide_index=True
                        )

        # ── VERIFICATION TABLE ────────────────────────────
        if not st.session_state.master_database.empty:
            st.divider()
            st.subheader("Data Verification Table")
            st.session_state.master_database = st.data_editor(
                st.session_state.master_database,
                num_rows="dynamic",
                use_container_width=True,
                key="data_verifier"
            )

        # ── CLOUD SYNC ────────────────────────────────────
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
                    st.session_state.master_database, n_dupes = remove_duplicates(
                        st.session_state.master_database
                    )
                    if n_dupes:
                        st.info(f"Removed {n_dupes} duplicate(s) before commit.")

                    final_cols = ["System_ID"] + [
                        c for c in expected_cols if c.lower() != "system_id"
                    ]
                    for col in final_cols:
                        if col not in st.session_state.master_database.columns:
                            st.session_state.master_database[col] = "N/A"
                    st.session_state.master_database = st.session_state.master_database[
                        [c for c in final_cols if c in st.session_state.master_database.columns]
                    ]

                    missing = (
                        st.session_state.master_database["System_ID"]
                        .astype(str).str.strip().str.lower().isin(INVALID_VALUES)
                    )
                    if missing.any():
                        st.error(f"{missing.sum()} record(s) missing System_ID.")
                    else:
                        with st.spinner("Merging and committing…"):
                            try:
                                push_to_sheet_merge(
                                    st.session_state.master_database,
                                    active_sheet_url, project_tab
                                )
                                st.toast("☁️ Commit successful.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Commit failed: {exc}")
                                logger.error("Commit error: %s", exc)

        with col_pull_btn:
            if st.button("⬇️ Pull from Cloud", use_container_width=True):
                if not active_sheet_url:
                    st.error("No Google Sheet URL configured.")
                else:
                    with st.spinner("Downloading…"):
                        try:
                            st.session_state.master_database = pull_from_sheet(
                                active_sheet_url, project_tab
                            )
                            st.toast("Local cache synchronised.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Pull failed: {exc}")

        with col_export:
            if not st.session_state.master_database.empty:
                csv_bytes = st.session_state.master_database.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export CSV",
                    data=csv_bytes,
                    file_name=f"{project_tab}_{datetime.utcnow().strftime('%Y%m%d')}.csv",
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

            placeholder_map = {
                "nan": "N/A", "None": "N/A", "NaN": "N/A",
                "none": "N/A", "null": "N/A", "": "N/A"
            }
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype(str).str.strip().replace(placeholder_map, regex=False)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col].replace("N/A", pd.NA), errors="ignore")

            all_cols = [c for c in df.columns if c != "System_ID"]

            st.markdown("#### 🔍 Search & Filter")
            s1, s2 = st.columns([2, 1])
            with s1:
                global_search = st.text_input(
                    "Global search", placeholder="Filter rows across all columns…"
                )
            with s2:
                filter_col = st.selectbox("Filter by column", ["— None —"] + all_cols)

            filtered_df = df.copy()
            if global_search.strip():
                mask = filtered_df.astype(str).apply(
                    lambda c: c.str.contains(global_search.strip(), case=False, na=False)
                ).any(axis=1)
                filtered_df = filtered_df[mask]

            if filter_col != "— None —":
                unique_vals   = sorted(
                    filtered_df[filter_col].dropna().astype(str).unique().tolist()
                )
                default_sel   = unique_vals[:10] if len(unique_vals) > 10 else unique_vals
                selected_vals = st.multiselect(
                    f"Values for '{filter_col}'", unique_vals, default=default_sel
                )
                if selected_vals:
                    filtered_df = filtered_df[
                        filtered_df[filter_col].astype(str).isin(selected_vals)
                    ]

            st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** records.")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 📊 Grouping & Counts")
            group_col = st.selectbox("Group by", ["— None —"] + all_cols, key="group_sel")
            if group_col != "— None —":
                group_counts         = (
                    filtered_df[group_col].astype(str).value_counts().reset_index()
                )
                group_counts.columns = [group_col, "Count"]
                group_counts         = group_counts[group_counts[group_col] != "N/A"]
                st.dataframe(group_counts, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 📈 Visualisation")
            v1, v2 = st.columns(2)
            with v1:
                x_axis = st.selectbox("Categorical Axis (X)", all_cols, key="x_sel")
            with v2:
                y_axis = st.selectbox("Numerical Axis (Y)", all_cols, key="y_sel")

            chart_type = st.radio("Chart Type", ["Bar", "Pie", "Scatter"], horizontal=True)

            try:
                chart_df = filtered_df[
                    ~filtered_df[x_axis].astype(str).isin(["N/A", "nan"])
                    & ~filtered_df[y_axis].astype(str).isin(["N/A", "nan"])
                ].copy()

                if chart_df.empty:
                    st.warning("No plottable data with the current filters.")
                else:
                    if chart_type == "Bar":
                        fig = px.bar(chart_df, x=x_axis, y=y_axis, color=x_axis,
                                     title=f"{y_axis} by {x_axis}")
                    elif chart_type == "Pie":
                        fig = px.pie(chart_df, names=x_axis,
                                     title=f"Distribution of {x_axis}")
                    else:
                        fig = px.scatter(chart_df, x=x_axis, y=y_axis, color=x_axis,
                                         title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Visualisation error: {exc}")
                logger.warning("Chart error: %s", exc)
