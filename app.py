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
st.set_page_config(page_title="CloudResearch", layout="wide", page_icon="🔬")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ============================================================
# CLOUDRESEARCH — PRECISION DARK UI SYSTEM
# Typography: IBM Plex Sans (body) + Syne (display)
# Palette: Deep navy base, electric teal accent, slate surfaces
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>

/* ── ROOT VARIABLES ───────────────────────────────────────── */
:root {
    --bg:          #07090F;
    --surface:     #0D1119;
    --panel:       #111620;
    --border:      #1E2535;
    --border-lit:  #2A3550;
    --accent:      #00D4AA;
    --accent-dim:  rgba(0, 212, 170, 0.12);
    --accent-glow: rgba(0, 212, 170, 0.25);
    --red:         #FF4B6E;
    --amber:       #FFB547;
    --blue:        #4B9FFF;
    --text-hi:     #EDF0F7;
    --text-md:     #8B95AE;
    --text-lo:     #4A5468;
    --font-sans:   'IBM Plex Sans', sans-serif;
    --font-display:'Syne', sans-serif;
    --font-mono:   'IBM Plex Mono', monospace;
    --radius:      6px;
    --radius-lg:   10px;
}

/* ── GLOBAL RESET ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg) !important;
    color: var(--text-hi) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── ANIMATED TOP STRIPE ──────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), #4B9FFF, var(--accent), transparent);
    background-size: 300% 100%;
    animation: stripe 4s linear infinite;
    z-index: 9999;
}
@keyframes stripe {
    0%   { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

/* ── SIDEBAR ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Sidebar logo header area */
[data-testid="stSidebar"] .stMarkdown h1 {
    font-family: var(--font-display) !important;
    color: var(--accent) !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    color: var(--text-md) !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── MAIN HEADINGS ────────────────────────────────────────── */
.stApp h1 {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-hi) !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
    line-height: 1.2 !important;
}

.stApp h2 {
    font-family: var(--font-display) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-md) !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    margin-bottom: 1rem !important;
    margin-top: 0.5rem !important;
}

.stApp h3 {
    font-family: var(--font-display) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
    margin-bottom: 0.5rem !important;
}

/* ── TABS ─────────────────────────────────────────────────── */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 1.5rem !important;
}

button[data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-md) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.2rem !important;
    letter-spacing: 0.03em !important;
    transition: color 0.2s !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--text-hi) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
}

[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

[data-baseweb="tab-border"] {
    background-color: var(--border) !important;
}

/* ── INPUTS & TEXT AREAS ──────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div:first-child {
    background-color: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
    outline: none !important;
}

[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-md) !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 0.3rem !important;
}

/* ── SELECTBOX ────────────────────────────────────────────── */
[data-baseweb="select"] * {
    background-color: var(--panel) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
}

[data-baseweb="popover"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius) !important;
}

[data-baseweb="menu"] li {
    font-size: 0.85rem !important;
}

[data-baseweb="menu"] li:hover {
    background-color: var(--accent-dim) !important;
}

/* ── BUTTONS ──────────────────────────────────────────────── */
[data-testid="stButton"] button {
    background-color: var(--panel) !important;
    color: var(--text-hi) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
}

[data-testid="stButton"] button:hover {
    background-color: var(--border-lit) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* Primary button */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #00C49A, #00A882) !important;
    color: #001F19 !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00D4AA, #00B990) !important;
    color: #001F19 !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Form submit button */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #00C49A, #00A882) !important;
    color: #001F19 !important;
    border: none !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem 1.2rem !important;
    text-transform: uppercase !important;
    transition: all 0.18s ease !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #00D4AA, #00B990) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background-color: transparent !important;
    color: var(--text-md) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    border-radius: var(--radius) !important;
    transition: all 0.18s !important;
}

[data-testid="stDownloadButton"] button:hover {
    border-color: var(--blue) !important;
    color: var(--blue) !important;
}

/* ── RADIO BUTTONS ────────────────────────────────────────── */
[data-testid="stRadio"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
}

[data-testid="stRadio"] [data-testid="stMarkdown"] p {
    font-size: 0.82rem !important;
}

/* Radio option container */
[data-testid="stRadio"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.18s !important;
}

[data-testid="stRadio"] > div > div:has(input:checked) {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── ALERT BOXES ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1rem !important;
}

/* Info */
div[data-testid="stAlert"][class*="info"] {
    background-color: rgba(75,159,255,0.08) !important;
    border-color: var(--blue) !important;
    color: #A0C4FF !important;
}

/* Success */
div[data-testid="stAlert"][class*="success"] {
    background-color: rgba(0,212,170,0.08) !important;
    border-color: var(--accent) !important;
    color: #80EDD9 !important;
}

/* Warning */
div[data-testid="stAlert"][class*="warning"] {
    background-color: rgba(255,181,71,0.08) !important;
    border-color: var(--amber) !important;
    color: #FFD59F !important;
}

/* Error */
div[data-testid="stAlert"][class*="error"] {
    background-color: rgba(255,75,110,0.08) !important;
    border-color: var(--red) !important;
    color: #FF9EB4 !important;
}

/* ── EXPANDER ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-md) !important;
    padding: 0.6rem 0.8rem !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--text-hi) !important;
    background: var(--panel) !important;
}

[data-testid="stExpander"] summary svg {
    fill: var(--text-lo) !important;
}

/* ── FILE UPLOADER ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--surface) !important;
    border: 1.5px dashed var(--border-lit) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

[data-testid="stFileUploader"] label {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
}

[data-testid="stFileUploader"] button {
    font-size: 0.78rem !important;
}

/* ── DATA EDITOR / TABLE ──────────────────────────────────── */
/* glide-data-grid renders on canvas — CSS cannot style cells.
   Solution: give the wrapper a light background so the grid's 
   internal dark text is always readable against it.           */
[data-testid="stDataEditor"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    background: #F8FAFC !important;
    padding: 0 !important;
}

[data-testid="stDataEditor"] > div {
    background: #F8FAFC !important;
}

/* The actual canvas wrapper */
[data-testid="stDataEditor"] .dvn-scroller,
[data-testid="stDataEditor"] canvas {
    background: #F8FAFC !important;
}

/* Column header strip above the canvas */
[data-testid="stDataEditor"] [role="columnheader"],
[data-testid="stDataEditor"] [role="gridcell"] {
    background: #F8FAFC !important;
    color: #1A202C !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

/* ── SPINNER ──────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
}

[data-testid="stSpinner"] svg {
    color: var(--accent) !important;
}

/* ── TOAST ────────────────────────────────────────────────── */
[data-testid="stToast"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    color: var(--text-hi) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}

/* ── DIVIDER ──────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── CAPTION & SMALL TEXT ─────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    font-family: var(--font-sans) !important;
    font-size: 0.72rem !important;
    color: var(--text-lo) !important;
    letter-spacing: 0.02em !important;
}

/* ── SIDEBAR SESSION STATUS ───────────────────────────────── */
[data-testid="stSidebar"] .stSuccess {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    border-radius: var(--radius) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 0.8rem !important;
}

/* ── FORM CONTAINER ───────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem !important;
}

/* ── SCROLLBAR ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-lo); }

/* ── METRIC CARDS ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.2rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-lo) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
}

/* ── CHECKBOX ─────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    font-size: 0.82rem !important;
    color: var(--text-md) !important;
    font-family: var(--font-sans) !important;
}

/* ── PLOTLY CHARTS ────────────────────────────────────────── */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* ── CUSTOM SECTION LABEL ─────────────────────────────────── */
.section-label {
    font-family: var(--font-sans);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-lo);
    margin-bottom: 0.75rem;
    margin-top: 0.5rem;
}

/* ── COLUMN LAYOUT TWEAKS ─────────────────────────────────── */
[data-testid="column"] {
    padding: 0 0.4rem !important;
}

/* ── MAIN CONTENT PADDING ─────────────────────────────────── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── TITLE BADGE ──────────────────────────────────────────── */
.cr-title-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.2rem;
}

.cr-badge {
    display: inline-block;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 0.4rem;
}

/* ── ID PILL STYLE ────────────────────────────────────────── */
.id-pill {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    background: var(--accent-dim);
    color: var(--accent);
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    border: 1px solid rgba(0,212,170,0.2);
    letter-spacing: 0.05em;
}

/* ── STATUS DOT ───────────────────────────────────────────── */
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    margin-right: 0.4rem;
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── SIDEBAR LOGO MARK ────────────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}

.sidebar-hexmark {
    width: 28px; height: 28px;
    background: var(--accent-dim);
    border: 1.5px solid var(--accent);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    color: var(--accent);
    font-weight: 700;
    font-family: var(--font-mono);
    flex-shrink: 0;
}

.sidebar-wordmark {
 
