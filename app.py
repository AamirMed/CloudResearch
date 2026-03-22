# ================================

# CLOUDRESEARCH (REFINED VERSION)

# ================================

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
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

# -------------------------------

# PAGE CONFIG

# -------------------------------

st.set_page_config(page_title="CloudResearch", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -------------------------------

# HELPERS

# -------------------------------

@st.cache_resource
def get_google_sheet_client():
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
return gspread.authorize(creds)

def generate_unique_id(existing_ids):
alphabet = string.ascii_uppercase + string.digits
for _ in range(10000):
new_id = f"CR-{''.join(random.choices(alphabet, k=4))}"
if new_id not in existing_ids:
return new_id
raise Exception("ID generation failed")

def convert_pdf_to_images(pdf_bytes):
image_list = []
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
for page in doc:
pix = page.get_pixmap(dpi=150)
image_list.append(pix.tobytes("jpeg"))
return image_list

def compress_image(image_bytes):
img = Image.open(io.BytesIO(image_bytes))
if img.mode != 'RGB':
img = img.convert('RGB')
img.thumbnail((2048, 2048))
output = io.BytesIO()
img.save(output, format="JPEG", quality=85)
return output.getvalue()

# -------------------------------

# PROMPT BUILDER (NEW CORE)

# -------------------------------

def build_prompt(user_prompt, abbreviations, extra_rules, anti_rules):
default_anti = "- Do not hallucinate values\n- Do not guess missing data"

```
return f"""
```

ROLE:
You are an expert clinical data extraction system.

USER INSTRUCTION:
{user_prompt}

ABBREVIATIONS:
{abbreviations if abbreviations.strip() else "None"}

CORE RULES:

* Use clinical reasoning
* Expand medical abbreviations
* Prefer structured data

EXTRA RULES:
{extra_rules if extra_rules.strip() else "None"}

ANTI-RULES:
{anti_rules if anti_rules.strip() else default_anti}

OUTPUT:
Return ONLY valid JSON. Use "N/A" if missing.
"""

# -------------------------------

# AI ENGINE

# -------------------------------

def blueprint_decoder(image_bytes, columns, model_choice):
prompt = build_prompt(
st.session_state.user_prompt,
st.session_state.abbreviations,
st.session_state.extra_rules,
st.session_state.anti_rules
)

```
full_prompt = f"""
```

{prompt}

REQUIRED COLUMNS:
[{columns}]

Return JSON array only.
"""

```
compressed = compress_image(image_bytes)

if "Gemini" in model_choice:
    model = genai.GenerativeModel(st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash"))
    response = model.generate_content(
        [full_prompt, Image.open(io.BytesIO(compressed))],
        generation_config={"response_mime_type": "application/json"}
    )
    return response.text.strip()
else:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    b64 = base64.b64encode(compressed).decode()
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    )
    return response.choices[0].message.content.strip()
```

# -------------------------------

# AUTH

# -------------------------------

def unlock_vault(d):
return {k: unlock_vault(v) if isinstance(v, dict) else v for k, v in d.items()}

try:
creds = unlock_vault(st.secrets["credentials"])
auth = stauth.Authenticate(
creds,
st.secrets["cookie"]["name"],
st.secrets["cookie"]["key"],
st.secrets["cookie"]["expiry_days"]
)
auth.login()
except Exception as e:
st.error(f"Auth error: {e}")
st.stop()

# -------------------------------

# MAIN APP

# -------------------------------

if st.session_state.get("authentication_status"):

```
st.title("CloudResearch")
st.caption("Clinical Data Extraction & Analysis")

username = st.session_state["username"]

# -------------------------------
# SIDEBAR (CLEANED)
# -------------------------------
with st.sidebar:

    st.success(f"Welcome, {st.session_state['name']}")
    auth.logout("Logout")

    st.divider()

    st.subheader("Extraction Engine")
    model_choice = st.selectbox("Model", ["Google Gemini", "Groq"])

    st.divider()

    st.subheader("Data Structure")

    if "schema_input" not in st.session_state:
        st.session_state.schema_input = "Age, Gender, Organism"

    st.session_state.schema_input = st.text_input(
        "Columns",
        st.session_state.schema_input
    )

    st.divider()

    # -------------------------------
    # PROMPT SYSTEM (NEW)
    # -------------------------------
    st.subheader("Extraction Setup")

    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = "Extract structured clinical data."

    st.session_state.user_prompt = st.text_area(
        "Prompt",
        st.session_state.user_prompt,
        height=100
    )

    # Quick buttons
    c1, c2, c3 = st.columns(3)
    if c1.button("Microbiology"): 
        st.session_state.user_prompt = "Extract organism and antibiotic sensitivity."
        st.rerun()
    if c2.button("Demographics"):
        st.session_state.user_prompt = "Extract age, gender, and demographics."
        st.rerun()
    if c3.button("Labs"):
        st.session_state.user_prompt = "Extract lab results."
        st.rerun()

    with st.expander("Advanced"):
        st.session_state.abbreviations = st.text_area(
            "Abbreviations",
            st.session_state.get("abbreviations",""),
            placeholder="DM → Diabetes Mellitus"
        )
        st.session_state.extra_rules = st.text_area(
            "Extra Rules",
            st.session_state.get("extra_rules","")
        )
        st.session_state.anti_rules = st.text_area(
            "Anti-Rules",
            st.session_state.get("anti_rules","")
        )

# -------------------------------
# MAIN WORKFLOW
# -------------------------------
uploaded = st.file_uploader("Upload Documents", type=["png","jpg","jpeg","pdf"], accept_multiple_files=True)

if st.button("Run Extraction", type="primary") and uploaded:

    ready = []
    for f in uploaded:
        if f.name.endswith(".pdf"):
            ready.extend(convert_pdf_to_images(f.getvalue()))
        else:
            ready.append(f.getvalue())

    data = []
    for img in ready:
        try:
            out = blueprint_decoder(img, st.session_state.schema_input, model_choice)
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                parsed = [parsed]
            data.extend(parsed)
        except:
            continue

    if data:
        df = pd.DataFrame(data)
        st.success("Extraction complete")
        st.dataframe(df, use_container_width=True)

        fig = px.histogram(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data extracted")
```

else:
st.warning("Login required")
