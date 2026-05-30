# =========================================================
# TALANG.IN - DATA SCIENCE ANALYTICS DASHBOARD v4.0
# GoFood NER Project | Streamlit
# =========================================================
# STRUKTUR BARU (sesuai saran A Moy):
#   1. Overview        - penjelasan project & alur data
#   2. Data Source     - list dataset + kategori utama/pendukung/eksplorasi
#   3. Data Cleaning   - before/after cleaning
#   4. EDA Data Utama  - grafik + filter di sini
#   5. Insight Data    - kesimpulan dari EDA
#   6. NER Dataset     - statistik training_data.json
#   7. A/B Testing     - simulasi reminder
#   8. Kesimpulan      - rangkuman akhir
# =========================================================

# ── 1. IMPORT & KONFIGURASI ────────────────────────────────
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import os
from collections import Counter

# Path otomatis - App.py ada di StreamlitData/, data ada di ../data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUT_DIR  = os.path.join(BASE_DIR, "..", "outputs")
ROOT_DIR = os.path.join(BASE_DIR, "..")

st.set_page_config(
    page_title="Talang.in Data Science Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. CSS STYLING ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400&display=swap');

/* ── FORCE LIGHT MODE & TEXT VISIBILITY ── */
html, body, .stApp, [class*="css"] { color-scheme: light !important; }
.stApp, .block-container { color: #1B4332 !important; }
.stApp p, .stApp div, .stApp span, .stApp li, .stApp ol, .stApp ul { color: #1B4332; }
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span { color: #1B4332 !important; }
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: #1B4332 !important; }
.streamlit-expanderHeader p, details summary p { color: #1B4332 !important; }
.stDataFrame *, div[data-testid="stDataFrame"] * { color: #1B4332 !important; }
.stSlider label, .stMultiSelect label, .stSelectbox label { color: #1B4332 !important; }
.stTabs [data-baseweb="tab"] p { color: #5A8A6A !important; }
.stTabs [aria-selected="true"] p { color: #1B4332 !important; }
div[data-testid="stAlert"] p { color: #1B4332 !important; }

/* ── EXCEPTION: header & sidebar tetap putih ── */
.page-header h1, .page-header h1 * { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
.page-header p, .page-header p *   { color: #B8DFC8 !important; -webkit-text-fill-color: #B8DFC8 !important; }
.page-header span, .header-badge   { color: #D4FFDC !important; }
.page-header b, .page-header strong { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span { color: #D4EDDA !important; }

/* ── ANIMATIONS ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(45,154,90,0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(45,154,90,0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* ── BASE ── */
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #EDF7EE 0%, #E3F2E4 40%, #EAF5EA 100%); min-height: 100vh; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1300px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071F16 0%, #0D3B2E 40%, #134E3A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #D4EDDA !important; }
section[data-testid="stSidebar"] label { color: #A8D5B5 !important; font-size: 13px !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── PAGE HEADER ── */
.page-header {
    background: linear-gradient(135deg, #0A3D27 0%, #145C38 50%, #1B6E44 100%);
    border-radius: 24px;
    padding: 40px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(13,74,48,0.3), 0 4px 16px rgba(0,0,0,0.1);
    animation: fadeInUp 0.6s ease;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(82,194,122,0.08);
    pointer-events: none;
}
.page-header h1 {
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.page-header p { color: #B8DFC8 !important; font-size: 14px; margin: 0; line-height: 1.8; }
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    color: #D4FFDC !important;
    font-size: 11px; font-weight: 700;
    padding: 5px 16px; border-radius: 20px;
    margin-bottom: 16px; text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ── SECTION HEADER ── */
.section-header {
    display: flex; align-items: center; gap: 14px;
    margin: 36px 0 8px 0; padding-bottom: 14px;
    border-bottom: 2px solid #C8E6C9;
    animation: fadeInLeft 0.4s ease;
}
.section-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: #2D9A5A;
    box-shadow: 0 4px 12px rgba(45,154,90,0.15);
}
.section-icon i { color: #2D9A5A !important; }
.section-title-text { font-size: 22px; font-weight: 800; color: #1B4332; margin: 0; letter-spacing: -0.3px; }
.section-desc { font-size: 14px; color: #6A9A7A; margin: 0 0 20px 0; line-height: 1.6; }

/* ── METRIC CARDS ── */
.metric-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 22px 24px;
    border: 1.5px solid #E0EFE0;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    box-shadow: 0 2px 12px rgba(27,67,50,0.06);
    animation: fadeInUp 0.5s ease both;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2D9A5A, #52C27A, #2D9A5A);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(27,67,50,0.12);
    border-color: #A5D6A7;
}
.metric-label {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
    color: #8AB89A; margin-bottom: 6px;
}
.metric-value {
    font-size: 28px; font-weight: 800;
    color: #1B4332; line-height: 1.1;
    animation: countUp 0.6s ease;
}
.metric-sub { font-size: 12px; color: #A8C8B0; margin-top: 6px; }
.metric-icon { font-size: 26px; margin-bottom: 12px; display: block; }

/* ── STAT CARDS (overview big numbers) ── */
.stat-card {
    background: linear-gradient(135deg, #FFFFFF, #F0FFF4);
    border-radius: 20px;
    padding: 24px;
    border: 1.5px solid #D4EDD8;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 12px rgba(45,154,90,0.07);
    animation: fadeInUp 0.5s ease both;
}
.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 40px rgba(45,154,90,0.15);
}
.stat-number {
    font-size: 36px; font-weight: 800;
    background: linear-gradient(135deg, #0D4A30, #2D9A5A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1; margin: 8px 0 4px;
}
.stat-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #8AB89A; }
.stat-sub { font-size: 12px; color: #A8C8B0; margin-top: 4px; }
.stat-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin: 0 auto 10px;
    box-shadow: 0 4px 12px rgba(45,154,90,0.15);
}

/* ── NER CARDS ── */
.ner-card {
    border-radius: 18px; padding: 22px 20px;
    text-align: center; height: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    animation: fadeInUp 0.5s ease both;
}
.ner-card:hover { transform: translateY(-4px) scale(1.02); box-shadow: 0 12px 32px rgba(0,0,0,0.12); }
.ner-label { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px; }
.ner-value { font-size: 32px; font-weight: 800; line-height: 1; margin-bottom: 6px; animation: countUp 0.8s ease; }

/* ── ALERT BOXES ── */
.alert-box {
    border-radius: 16px; padding: 16px 20px;
    margin-bottom: 14px; font-size: 14px; line-height: 1.7;
    animation: fadeInUp 0.4s ease;
    transition: all 0.2s ease;
}
.alert-box:hover { transform: translateX(3px); }
.alert-info    { background: linear-gradient(135deg, #EBF8EE, #F0FFF4); border-left: 4px solid #2D9A5A; color: #1B4332 !important; box-shadow: 0 2px 12px rgba(45,154,90,0.08); }
.alert-success { background: linear-gradient(135deg, #E8F5E9, #F1FBF1); border-left: 4px solid #2E7D32; color: #1B4332 !important; box-shadow: 0 2px 12px rgba(46,125,50,0.08); }
.alert-warning { background: linear-gradient(135deg, #FFF8E1, #FFFDE7); border-left: 4px solid #F9A825; color: #5D4037 !important; box-shadow: 0 2px 12px rgba(249,168,37,0.08); }
.alert-box * { color: inherit !important; }
.alert-title { font-weight: 700; font-size: 14px; margin-bottom: 6px; }

/* ── BADGES ── */
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.3px; }
.badge-green  { background: linear-gradient(135deg, #C8E6C9, #A5D6A7); color: #1B5E20 !important; }
.badge-yellow { background: linear-gradient(135deg, #FFF9C4, #FFF176); color: #795548 !important; }
.badge-red    { background: linear-gradient(135deg, #FFCDD2, #EF9A9A); color: #B71C1C !important; }
.badge-blue   { background: linear-gradient(135deg, #BBDEFB, #90CAF9); color: #0D47A1 !important; }
.badge-purple { background: linear-gradient(135deg, #E1BEE7, #CE93D8); color: #4A148C !important; }

/* ── DATA CHIPS ── */
.data-chip { display: inline-block; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 700; margin: 4px; border: 1.5px solid; transition: all 0.2s ease; }
.data-chip:hover { transform: scale(1.05); }
.chip-utama     { background: linear-gradient(135deg, #E8F5E9, #F0FFF4); color: #1B5E20; border-color: #81C784; }
.chip-pendukung { background: linear-gradient(135deg, #E3F2FD, #EFF8FF); color: #0D47A1; border-color: #64B5F6; }
.chip-eksplorasi{ background: linear-gradient(135deg, #FFF8E1, #FFFDE7); color: #795548; border-color: #FFD54F; }

/* ── PIPELINE STEPS ── */
.pipeline-card {
    background: #FFFFFF; border-radius: 18px; padding: 22px 24px;
    border: 1.5px solid #E0EFE0; height: 100%;
    box-shadow: 0 4px 16px rgba(27,67,50,0.06);
    animation: fadeInUp 0.5s ease;
}
.pipeline-step { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 16px; }
.step-num {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #1B4332, #2D9A5A);
    color: white; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(45,154,90,0.3);
}
.step-text { font-size: 13px; color: #2D5A3D; line-height: 1.7; }
.step-label { font-weight: 700; color: #1B4332; font-size: 13px; }

/* ── A/B TESTING CARDS ── */
.ab-result-card {
    background: #FFFFFF; border-radius: 18px; padding: 20px 16px;
    border: 1.5px solid #E0EFE0; text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.ab-result-card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.1); }
.ab-variant-a { border-top: 4px solid #1565C0; }
.ab-variant-b { border-top: 4px solid #2E7D32; }
.ab-stat { font-size: 28px; font-weight: 800; line-height: 1.1; margin: 8px 0 4px 0; }
.ab-label { font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #8AB89A; }

/* ── DATA SOURCE CARDS ── */
.ds-card {
    background: #FFFFFF; border-radius: 18px; padding: 18px 20px;
    border: 1.5px solid #E0EFE0; margin-bottom: 10px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 10px rgba(27,67,50,0.05);
    animation: fadeInUp 0.4s ease both;
}
.ds-card:hover {
    transform: translateX(6px);
    box-shadow: 0 8px 24px rgba(27,67,50,0.1);
    border-color: #A5D6A7;
}

/* ── CHECK ITEMS ── */
.check-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #EBF5EB;
    transition: all 0.2s ease;
}
.check-item:hover { background: #F8FFF8; padding-left: 8px; border-radius: 8px; }

/* ── FOOTER ── */
.footer-area {
    background: linear-gradient(135deg, #FFFFFF, #F5FFF6);
    border-radius: 20px; padding: 28px 32px;
    text-align: center; border: 1.5px solid #D4EDD8;
    margin-top: 48px;
    box-shadow: 0 4px 20px rgba(27,67,50,0.08);
}
.footer-title { font-size: 16px; font-weight: 800; color: #1B4332; margin-bottom: 6px; letter-spacing: -0.3px; }
.footer-area p { color: #6A9A7A !important; font-size: 13px; margin: 4px 0; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #81C784, #2D9A5A); border-radius: 4px; }

/* ── PROGRESS BAR ── */
.stProgress > div > div { background: linear-gradient(90deg, #2D9A5A, #52C27A, #81C784) !important; border-radius: 8px !important; }
.stProgress > div { background: #D4EDD8 !important; border-radius: 8px !important; height: 10px !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background: #E8F5E9; border-radius: 14px; padding: 4px; box-shadow: inset 0 2px 6px rgba(27,67,50,0.06); }
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #5A8A6A; font-weight: 600; font-size: 14px; transition: all 0.2s ease; }
.stTabs [aria-selected="true"] { background: #FFFFFF !important; color: #1B4332 !important; box-shadow: 0 2px 8px rgba(27,67,50,0.1) !important; }

/* ── SIDEBAR RADIO ── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05); border-radius: 10px;
    padding: 8px 14px !important; margin-bottom: 4px;
    font-size: 13px !important; color: #C8E6C9 !important;
    display: block; width: 100%;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.1);
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(82,194,122,0.15) !important;
    border-left: 3px solid #52C27A;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── MULTISELECT & INPUT ── */
.stMultiSelect [data-baseweb="tag"] { background: #C8E6C9 !important; border-radius: 8px !important; }
div[data-testid="stMetric"] { background: #FFF; border-radius: 16px; padding: 16px; border: 1.5px solid #E0EFE0; }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
""", unsafe_allow_html=True)


# ── 3. LOAD & CACHE DATA ───────────────────────────────────
@st.cache_data
def load_gofood():
    df = pd.read_csv(os.path.join(DATA_DIR, "gofood_dataset.csv"))
    missing_before   = df.isnull().sum().sum()
    duplicate_before = int(df.duplicated().sum())
    df['discount_price'] = df['discount_price'].fillna(0)
    df_clean = df.drop_duplicates()
    missing_after = int(df_clean.isnull().sum().sum())
    return df, df_clean, int(missing_before), duplicate_before, missing_after

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def compute_ner_stats():
    training_data  = load_json(os.path.join(OUT_DIR, "training_data.json"))
    entity_counts  = Counter()
    item_counter   = Counter()
    person_counter = Counter()
    person_per_rec = []
    for rec in training_data:
        n_person = 0
        for ent in rec.get("entities", []):
            lbl = ent["label"]
            if lbl and not lbl.startswith("[") and lbl.strip():
                entity_counts[lbl] += 1
            if lbl == "ITEM":   item_counter[ent["text"].lower()] += 1
            if lbl == "PERSON": person_counter[ent["text"]] += 1; n_person += 1
        person_per_rec.append(n_person)
    return entity_counts, item_counter, person_counter, person_per_rec

df_raw, df, missing_before, duplicate_before, missing_after = load_gofood()
training_data  = load_json(os.path.join(OUT_DIR, "training_data.json"))
fixed_dataset  = load_json(os.path.join(ROOT_DIR, "dataset_fixed.json"))
templates      = load_json(os.path.join(OUT_DIR, "talangin_synthetic_templates.json"))
entity_counts, item_counter, person_counter, person_per_rec = compute_ner_stats()

total_entities   = sum(v for k, v in entity_counts.items() if k.strip() and not k.startswith('['))
total_merchant   = df['merchant_name'].nunique()
total_area       = df['merchant_area'].nunique()
avg_price        = df['price'].mean()
kategori_terbanyak = df['category'].value_counts().idxmax()
merchant_aktif   = df['merchant_name'].value_counts().idxmax()

health_score  = 100
health_issues = []
if duplicate_before > 0:
    health_score -= 20
    health_issues.append(f"{duplicate_before:,} duplikat ditemukan - sudah dibersihkan dengan drop_duplicates()")
if missing_before > 0:
    health_score -= 20
    health_issues.append(f"{missing_before:,} missing value ditemukan - sudah ditangani dengan fillna(0)")

health_color  = "#2E7D32" if health_score >= 80 else "#F57F17" if health_score >= 60 else "#C62828"
health_status = "Sangat Baik" if health_score >= 80 else "Perlu Perhatian" if health_score >= 60 else "Kritis"


# ── 4. SIDEBAR ─────────────────────────────────────────────
SECTIONS = [
    ("overview",      "fa-house",           "Overview",        "#52C27A"),
    ("data_source",   "fa-database",        "Data Source",     "#64B5F6"),
    ("data_cleaning", "fa-broom",           "Data Cleaning",   "#FFD54F"),
    ("eda",           "fa-chart-bar",       "EDA Data Utama",  "#CE93D8"),
    ("insight",       "fa-lightbulb",       "Insight Data",    "#FF8A65"),
    ("ner_dataset",   "fa-robot",           "NER Dataset",     "#4FC3F7"),
    ("ab_testing",    "fa-flask",           "A/B Testing",     "#A5D6A7"),
    ("kesimpulan",    "fa-clipboard-check", "Kesimpulan",      "#F48FB1"),
]

with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-logo {
        text-align:center; padding: 16px 0 20px;
        animation: fadeInUp 0.5s ease;
    }
    .sidebar-logo-icon {
        width: 56px; height: 56px; margin: 0 auto 12px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(82,194,122,0.25), rgba(255,255,255,0.1));
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; color: #FFF;
        border: 1.5px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    .sidebar-title { font-size: 18px; font-weight: 800; color: #FFF; letter-spacing: -0.3px; }
    .sidebar-sub   { font-size: 11px; color: #6BA882; margin-top: 3px; font-weight: 600; }
    .nav-badge {
        display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; margin-right: 4px; vertical-align: middle;
    }
    .info-card {
        background: linear-gradient(135deg, rgba(82,194,122,0.1), rgba(255,255,255,0.05));
        border: 1px solid rgba(82,194,122,0.2);
        border-radius: 14px; padding: 14px 16px;
    }
    .info-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 12px;
    }
    .info-row:last-child { border-bottom: none; }
    .info-key { color: #7ABFA0; }
    .info-val { color: #FFF; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon"><i class="fa-solid fa-wallet"></i></div>
        <div class="sidebar-title">Talang.in</div>
        <div class="sidebar-sub">
            <i class="fa-solid fa-circle" style="font-size:6px;color:#52C27A;vertical-align:middle;"></i>
            &nbsp;Data Science Dashboard v4.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);margin:4px 0 16px;"></div>', unsafe_allow_html=True)

    selected_section = st.radio(
        "Navigasi",
        options=[s[0] for s in SECTIONS],
        format_func=lambda k: next(f"  {s[2]}" for s in SECTIONS if s[0] == k),
        label_visibility="collapsed"
    )

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    # Info stats card
    st.markdown(f"""
    <div class="info-card">
        <div style="font-size:10px;color:#52C27A;font-weight:800;text-transform:uppercase;
        letter-spacing:1px;margin-bottom:10px;">
            <i class="fa-solid fa-chart-pie"></i> &nbsp;Statistik Project
        </div>
        <div class="info-row">
            <span class="info-key"><i class="fa-solid fa-database" style="width:14px;"></i> Data bersih</span>
            <span class="info-val">{len(df):,}</span>
        </div>
        <div class="info-row">
            <span class="info-key"><i class="fa-solid fa-robot" style="width:14px;"></i> Data latih</span>
            <span class="info-val">{len(training_data):,}</span>
        </div>
        <div class="info-row">
            <span class="info-key"><i class="fa-solid fa-tags" style="width:14px;"></i> Entitas</span>
            <span class="info-val">{total_entities:,}</span>
        </div>
        <div class="info-row">
            <span class="info-key"><i class="fa-solid fa-heart-pulse" style="width:14px;"></i> Health</span>
            <span class="info-val" style="color:{'#52C27A' if health_score >= 80 else '#FFD54F'};">{health_score}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);margin:16px 0 12px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-size:10px;color:#4A7A60;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Powered by</div>
        <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.08);color:#A8D5B5;font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;border:1px solid rgba(255,255,255,0.1);">GoFood</span>
            <span style="background:rgba(255,255,255,0.08);color:#A8D5B5;font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;border:1px solid rgba(255,255,255,0.1);">Gemma LLM</span>
            <span style="background:rgba(255,255,255,0.08);color:#A8D5B5;font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;border:1px solid rgba(255,255,255,0.1);">Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 5. HELPER FUNCTIONS ────────────────────────────────────
def section_header(icon, title, desc):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid {icon}"></i></div>
        <p class="section-title-text">{title}</p>
    </div>
    <p class="section-desc">{desc}</p>
    """, unsafe_allow_html=True)

def metric_card(col, icon_html, label, value, sub, font_size="26px"):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">{icon_html}</span>
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:{font_size};">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

def plotly_layout(fig, title, height=430):
    fig.update_layout(
        height=height, coloraxis_showscale=False,
        title=dict(text=title, font=dict(size=14, color='#1B4332')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=10, r=10)
    )
    return fig


# ── 6. HEADER UTAMA ────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="header-badge">
        <i class="fa-solid fa-robot"></i> NER | GoFood | Data Science | Analytics
    </div>
    <h1><i class="fa-solid fa-wallet"></i> Talang.in - Data Science Dashboard</h1>
    <p>Dashboard ini mendokumentasikan proses Data Science untuk mendukung fitur
    <b>AI Smart Transaction Input</b> pada aplikasi <b>Talang.in</b>.
    Dataset GoFood digunakan sebagai sumber referensi menu, harga, dan kategori
    untuk membangun training data NER.
    Fokus: <b>data preparation, EDA, preprocessing, dan kesiapan data latih.</b></p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 7. ROUTER SECTION
# ══════════════════════════════════════════════════════════

# ── OVERVIEW ──────────────────────────────────────────────
if selected_section == "overview":
    section_header("house", "Overview",
        "Penjelasan singkat project Talang.in, tujuan dashboard, dan alur data secara umum.")

    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-wallet"></i> Apa itu Talang.in?</div>
        Talang.in adalah aplikasi manajemen keuangan grup berbasis NLP. Pengguna cukup mengetik
        kalimat tagihan dalam Bahasa Indonesia, dan sistem AI akan otomatis mengekstrak siapa
        yang membayar, item apa, berapa harganya, dan untuk berapa orang.
        Contoh: <b>"Ayu bayar pizza 90k untuk Raka dan Nina"</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-success">
        <div class="alert-title"><i class="fa-solid fa-bullseye"></i> Tujuan Dashboard Data Science</div>
        Dashboard ini mendokumentasikan seluruh proses Data Science mulai dari pengumpulan data,
        pembersihan, eksplorasi, hingga menghasilkan dataset final yang siap digunakan oleh tim
        AI Engineer untuk melatih model NER (Named Entity Recognition).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### <i class='fa-solid fa-diagram-project'></i> Alur Data Secara Umum", unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-card">
        <div class="pipeline-step">
            <div class="step-num">1</div>
            <div class="step-text">
                <span class="step-label"><i class="fa-solid fa-file-import"></i> Pengumpulan Data</span><br>
                6 dataset dikumpulkan dari sumber publik - GoFood, nama Indonesia, data makanan, alergen, nutrisi, steakhouse
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">2</div>
            <div class="step-text">
                <span class="step-label"><i class="fa-solid fa-broom"></i> Cleaning & Preprocessing</span><br>
                Data utama (GoFood) dibersihkan - hapus duplikat, tangani missing value, normalisasi teks
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">3</div>
            <div class="step-text">
                <span class="step-label"><i class="fa-solid fa-chart-bar"></i> EDA & Analisis</span><br>
                Eksplorasi distribusi kategori, harga, merchant, dan area untuk memahami karakteristik data
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">4</div>
            <div class="step-text">
                <span class="step-label"><i class="fa-solid fa-code"></i> Generate Template Kalimat</span><br>
                Nama menu dari GoFood + nama orang digunakan sebagai bahan template kalimat transaksi sintetis
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">5</div>
            <div class="step-text">
                <span class="step-label"><i class="fa-solid fa-robot"></i> Dataset Final NER</span><br>
                training_data.json - berisi kalimat transaksi + label entity PERSON, ITEM, PRICE, MULTIPLIER
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    cols = st.columns(4)
    cards = [
        ('<i class="fa-solid fa-database"></i>',  "GoFood Raw",     f"{len(df_raw):,}",        "data mentah"),
        ('<i class="fa-solid fa-broom"></i>',      "Setelah Cleaning", f"{len(df):,}",          "data bersih"),
        ('<i class="fa-solid fa-robot"></i>',      "Data Latih NER", f"{len(training_data):,}", "records siap pakai"),
        ('<i class="fa-solid fa-tags"></i>',       "Total Entitas",  f"{total_entities:,}",     "PERSON, ITEM, PRICE, MULT"),
    ]
    for col, (icon, label, val, sub) in zip(cols, cards):
        metric_card(col, icon, label, val, sub)


# ── DATA SOURCE ───────────────────────────────────────────
elif selected_section == "data_source":
    section_header("database", "Data Source",
        "Daftar dataset yang digunakan, fungsi masing-masing, dan kategorinya: Utama, Pendukung, atau Eksplorasi.")

    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Kategorisasi Dataset</div>
        Dataset dibagi menjadi tiga kategori sesuai perannya dalam project:
        <span class="data-chip chip-utama">Data Utama</span> dipakai langsung untuk analisis &amp; sumber variasi NER.
        <span class="data-chip chip-pendukung">Data Pendukung</span> menambah variasi item &amp; nama.
        <span class="data-chip chip-eksplorasi">Data Eksplorasi</span> digunakan untuk riset awal, tidak masuk model utama.
    </div>
    """, unsafe_allow_html=True)

    datasets = [
        ("fa-store",        "gofood_dataset.csv",        "45.195",  "chip-utama",      "Data Utama",
         "Sumber utama nama produk/menu, harga, kategori, dan area merchant. Digunakan untuk analisis EDA dan sebagai sumber variasi entity ITEM dan PRICE pada training data NER."),
        ("fa-id-card",      "indonesian-names.csv",      "1.960",   "chip-utama",      "Data Utama",
         "Kumpulan nama orang Indonesia. Digunakan sebagai sumber variasi entity PERSON dalam template kalimat transaksi."),
        ("fa-bowl-food",    "indonesian_food.csv",       "1.273",   "chip-pendukung",  "Data Pendukung",
         "Nama-nama makanan Indonesia. Menambah variasi item menu di luar data GoFood agar model lebih general."),
        ("fa-utensils",     "Steakhouse_dataset.csv",    "150",     "chip-pendukung",  "Data Pendukung",
         "Menu restoran steakhouse Indonesia. Menambah variasi jenis makanan berat untuk entity ITEM."),
        ("fa-flask",        "alergen_dataset.csv",       "100.000", "chip-eksplorasi", "Data Eksplorasi",
         "Data label alergen pangan Indonesia. Digunakan pada tahap eksplorasi awal untuk menambah variasi nama produk, namun bukan data utama model NER."),
        ("fa-apple-whole",  "nutrition.csv",             "1.345",   "chip-eksplorasi", "Data Eksplorasi",
         "Informasi gizi makanan. Digunakan pada eksplorasi awal. Lebih relevan jika project memiliki fitur rekomendasi makanan atau filter nutrisi."),
    ]

    for icon, name, rows, chip_cls, chip_label, desc in datasets:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;padding:16px 20px;">
            <div style="display:flex;align-items:flex-start;gap:16px;">
                <div style="width:40px;height:40px;background:#E8F5E9;border-radius:10px;display:flex;
                align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">
                    <i class="fa-solid {icon}"></i></div>
                <div style="flex:1;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="font-size:14px;font-weight:700;color:#1B4332;">{name}</span>
                        <span class="data-chip {chip_cls}">{chip_label}</span>
                        <span class="badge badge-blue">{rows} baris</span>
                    </div>
                    <div style="font-size:13px;color:#2D5A3D;line-height:1.7;">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-success" style="margin-top:8px;">
        <div class="alert-title"><i class="fa-solid fa-file-export"></i> Output Akhir Data Science</div>
        Dari semua dataset di atas, dihasilkan <b>training_data.json</b> berisi kalimat transaksi
        berlabel entity - ini adalah file yang diserahkan ke tim AI Engineer untuk melatih model NER.
    </div>
    """, unsafe_allow_html=True)


# ── DATA CLEANING ─────────────────────────────────────────
elif selected_section == "data_cleaning":
    section_header("broom", "Data Cleaning",
        "Proses pembersihan data GoFood - jumlah data sebelum dan sesudah cleaning, penanganan missing value dan duplikasi.")

    pct_buang = (len(df_raw) - len(df)) / len(df_raw) * 100

    cols = st.columns(4)
    cards = [
        ('<i class="fa-solid fa-database"></i>',    "Data Mentah",      f"{len(df_raw):,}",  "sebelum cleaning"),
        ('<i class="fa-solid fa-circle-check"></i>', "Data Bersih",      f"{len(df):,}",      "setelah cleaning"),
        ('<i class="fa-solid fa-copy"></i>',         "Duplikat Dihapus", f"{duplicate_before:,}", "baris terbuang"),
        ('<i class="fa-solid fa-circle-xmark"></i>', "Missing Value",    f"{missing_before:,}",   "cell kosong awal"),
    ]
    for col, (icon, label, val, sub) in zip(cols, cards):
        metric_card(col, icon, label, val, sub)

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="alert-box {'alert-warning' if missing_before > 0 else 'alert-success'}">
            <div class="alert-title"><i class="fa-solid fa-table-cells"></i> Penanganan Missing Value</div>
            Ditemukan <b>{missing_before:,} cell kosong</b> pada data mentah.<br>
            Kolom <b>discount_price</b> diisi dengan <code>fillna(0)</code> karena produk tanpa
            diskon memang tidak memiliki nilai - bukan error data.<br>
            Setelah cleaning: <b>{missing_after:,} missing value</b> tersisa.
        </div>
        """, unsafe_allow_html=True)

        miss_detail = df_raw.isnull().sum().reset_index()
        miss_detail.columns = ['Kolom', 'Jumlah Missing']
        miss_detail['Status'] = miss_detail['Jumlah Missing'].apply(
            lambda x: 'Bersih' if x == 0 else f'{x:,} missing')
        st.dataframe(miss_detail, use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f"""
        <div class="alert-box {'alert-warning' if duplicate_before > 0 else 'alert-success'}">
            <div class="alert-title"><i class="fa-solid fa-copy"></i> Penanganan Duplikasi</div>
            Ditemukan <b>{duplicate_before:,} baris duplikat</b> pada data mentah.<br>
            Dihapus menggunakan <code>drop_duplicates()</code>.<br>
            Data bersih: <b>{len(df):,} baris</b> dari {len(df_raw):,}
            ({pct_buang:.1f}% data terbuang).
        </div>
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-list-check"></i> Kolom yang Digunakan</div>
            <b>Dipakai:</b> merchant_name, merchant_area, category, product, price, discount_price<br>
            <b>Fitur baru:</b> isDiscount (boolean), category_encoded (label encoding)<br>
            <b>Tidak dipakai:</b> display, description (terlalu panjang untuk NER)
        </div>
        """, unsafe_allow_html=True)

        before_after = pd.DataFrame({
            'Kondisi': ['Sebelum Cleaning', 'Setelah Cleaning'],
            'Jumlah Baris': [len(df_raw), len(df)]
        })
        fig = px.bar(before_after, x='Kondisi', y='Jumlah Baris', color='Kondisi',
                     text_auto=True, color_discrete_map={
                         'Sebelum Cleaning': '#F9A825',
                         'Setelah Cleaning': '#2D9A5A'
                     }, template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Jumlah Data Sebelum vs Sesudah Cleaning', height=300),
                        use_container_width=True)


# ── EDA DATA UTAMA ────────────────────────────────────────
elif selected_section == "eda":
    section_header("chart-bar", "EDA Data Utama",
        "Eksplorasi data GoFood - distribusi kategori, harga, merchant, dan area. Filter tersedia di halaman ini.")

    # Filter di halaman EDA, bukan di sidebar
    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-filter"></i> Filter Data Eksplorasi</div>
        Gunakan filter di bawah untuk mengeksplorasi data berdasarkan kategori dan area merchant.
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        kategori_filter = st.multiselect(
            "Filter Kategori", options=df['category'].unique(), default=df['category'].unique())
    with col_f2:
        area_filter = st.multiselect(
            "Filter Area Merchant", options=df['merchant_area'].unique(), default=df['merchant_area'].unique())

    filtered_df = df[df['category'].isin(kategori_filter) & df['merchant_area'].isin(area_filter)]

    st.markdown(f"""
    <div style="background:#E8F5E9;border-radius:12px;padding:12px 18px;margin-bottom:16px;
    font-size:13px;color:#1B4332;">
        <i class="fa-solid fa-circle-info"></i>
        Menampilkan <b>{len(filtered_df):,} produk</b> dari
        <b>{filtered_df['merchant_name'].nunique():,} merchant</b> di
        <b>{filtered_df['merchant_area'].nunique()} area</b>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribusi Kategori", "Top Merchant", "Distribusi Harga", "Distribusi Area"
    ])

    with tab1:
        kategori_data = filtered_df['category'].value_counts().reset_index()
        kategori_data.columns = ['Kategori', 'Jumlah']
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(kategori_data.head(15), x='Kategori', y='Jumlah', color='Jumlah',
                         text_auto=True, color_continuous_scale=['#A8D5B5', '#2D9A5A', '#0D4A30'],
                         template='plotly_white')
            st.plotly_chart(plotly_layout(fig, 'Top 15 Kategori Produk'), use_container_width=True)
        with col2:
            fig = px.pie(kategori_data.head(8), names='Kategori', values='Jumlah',
                         color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.45)
            st.plotly_chart(plotly_layout(fig, 'Proporsi Top 8 Kategori'), use_container_width=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-lightbulb"></i> Insight Kategori</div>
            Kategori dominan adalah <b>{kategori_terbanyak}</b>. Keragaman kategori ini
            memperkaya variasi entity <b>ITEM</b> pada training data NER sehingga model
            dapat mengenali berbagai jenis menu makanan dan minuman.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        merchant_data = filtered_df['merchant_name'].value_counts().head(10).reset_index()
        merchant_data.columns = ['Merchant', 'Jumlah Produk']
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(merchant_data.sort_values('Jumlah Produk'), x='Jumlah Produk', y='Merchant',
                         orientation='h', color='Jumlah Produk', text_auto=True,
                         color_continuous_scale=['#A8D5B5', '#2D9A5A', '#0D4A30'], template='plotly_white')
            st.plotly_chart(plotly_layout(fig, 'Top 10 Merchant berdasarkan Jumlah Produk', height=400),
                            use_container_width=True)
        with col2:
            fig = px.pie(merchant_data, names='Merchant', values='Jumlah Produk',
                         color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.5)
            st.plotly_chart(plotly_layout(fig, 'Porsi Top 10 Merchant', height=400),
                            use_container_width=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-store"></i> Insight Merchant</div>
            Terdapat <b>{filtered_df['merchant_name'].nunique():,} merchant unik</b>.
            Merchant dengan produk terbanyak: <b>{merchant_aktif}</b>.
            Banyaknya merchant memastikan variasi nama produk yang kaya untuk entity ITEM.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered_df[filtered_df['price'] < 200000], x='price',
                               nbins=50, color_discrete_sequence=['#2D9A5A'], template='plotly_white')
            st.plotly_chart(plotly_layout(fig, 'Distribusi Harga Produk (< Rp 200.000)', height=350),
                            use_container_width=True)
        with col2:
            price_by_cat = filtered_df.groupby('category')['price'].mean().sort_values(ascending=False).head(10).reset_index()
            price_by_cat.columns = ['Kategori', 'Rata-rata Harga']
            fig = px.bar(price_by_cat, x='Rata-rata Harga', y='Kategori', orientation='h',
                         color='Rata-rata Harga', text_auto='.0f',
                         color_continuous_scale=['#A8D5B5', '#2D9A5A', '#0D4A30'], template='plotly_white')
            st.plotly_chart(plotly_layout(fig, 'Rata-rata Harga per Kategori (Top 10)', height=350),
                            use_container_width=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-coins"></i> Insight Harga</div>
            Rata-rata harga produk: <b>Rp {avg_price:,.0f}</b>.
            Rentang harga yang luas (murah hingga mahal) memberikan variasi yang baik
            untuk entity <b>PRICE</b> dalam training data NER.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        area_data = filtered_df['merchant_area'].value_counts().reset_index()
        area_data.columns = ['Area', 'Jumlah Produk']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=area_data['Area'], y=area_data['Jumlah Produk'],
            marker_color='#2D9A5A', text=area_data['Jumlah Produk'],
            textposition='outside'
        ))
        fig.update_layout(height=380,
                          title=dict(text=f'Jumlah Produk per Area ({filtered_df["merchant_area"].nunique()} Area)',
                                     font=dict(size=14, color='#1B4332')),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-location-dot"></i> Insight Area</div>
            Data mencakup <b>{filtered_df['merchant_area'].nunique()} area</b> di Indonesia.
            Area dengan volume produk tinggi menghasilkan variasi nama menu yang lebih kaya
            untuk keperluan training data NER.
        </div>
        """, unsafe_allow_html=True)


# ── INSIGHT DATA ──────────────────────────────────────────
elif selected_section == "insight":
    section_header("lightbulb", "Insight Data",
        "Ringkasan temuan dari EDA dan hubungannya dengan kebutuhan AI Smart Transaction Input.")

    insights = [
        ("store",       "Keragaman Merchant & Menu",
         f"Dataset mencatat <b>{len(df):,} produk</b> dari <b>{total_merchant:,} merchant</b> di <b>{total_area} area</b>. Keragaman ini memastikan variasi entity ITEM yang kaya dalam training data NER - model dapat mengenali berbagai jenis nama menu dari seluruh Indonesia."),
        ("coins",       "Variasi Format Harga",
         f"Rata-rata harga produk <b>Rp {avg_price:,.0f}</b> dengan rentang luas. Variasi harga ini digunakan untuk menghasilkan variasi penulisan entity PRICE dalam kalimat transaksi, mulai dari format <b>Rp 20.000</b> hingga <b>20k</b> dan <b>dua puluh ribu</b>."),
        ("tag",         "Kategori Dominan",
         f"Kategori paling banyak: <b>{kategori_terbanyak}</b>. Distribusi kategori yang beragam memastikan training data NER tidak bias ke satu jenis makanan saja, sehingga model lebih general dalam mengenali berbagai item menu."),
        ("user",        "Kebutuhan Entity PERSON",
         f"Data nama Indonesia (<b>1.960 nama unik</b>) dikombinasikan dengan data GoFood untuk membentuk kalimat transaksi realistis. Satu kalimat bisa mengandung <b>1 hingga 8 nama orang</b> sesuai skenario split bill grup."),
        ("robot",       "Hubungan dengan AI Smart Input",
         f"Semua insight di atas bermuara pada satu output: <b>training_data.json</b> berisi <b>{len(training_data):,} kalimat transaksi</b> berlabel entity. File ini diserahkan ke tim AI Engineer sebagai bahan latih model NER untuk fitur Smart Transaction Input."),
        ("triangle-exclamation", "Catatan Data Eksplorasi",
         "Dataset alergen dan nutrisi tidak masuk ke model NER utama karena isinya berkaitan dengan kandungan gizi dan alergen - lebih relevan jika Talang.in mengembangkan fitur rekomendasi makanan atau filter alergi di masa depan."),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, body) in enumerate(insights):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="alert-box alert-info">
                <div class="alert-title"><i class="fa-solid fa-{icon}"></i> {title}</div>
                {body}
            </div>
            """, unsafe_allow_html=True)


# ── NER DATASET ───────────────────────────────────────────
elif selected_section == "ner_dataset":
    section_header("robot", "NER Dataset",
        "Statistik dataset final training_data.json - entity PERSON, ITEM, PRICE, MULTIPLIER siap untuk tim AI Engineer.")

    st.markdown("""
    <div class="alert-box alert-warning">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Status Dataset NER</div>
        Dataset ini <b>siap diserahkan ke tim AI Engineer</b>. Proses training model dan evaluasi
        (precision, recall, F1-score) dilakukan oleh tim AI, bukan bagian dari scope Data Science.
        Dashboard ini hanya menampilkan statistik kesiapan data latih.
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    ner_cards = [
        ('<i class="fa-solid fa-user"></i>',    "PERSON",     entity_counts.get('PERSON', 0),     "nama orang",  "#1B5E20", "#E8F5E9"),
        ('<i class="fa-solid fa-utensils"></i>', "ITEM",       entity_counts.get('ITEM', 0),       "nama menu",   "#0D47A1", "#E3F2FD"),
        ('<i class="fa-solid fa-coins"></i>',    "PRICE",      entity_counts.get('PRICE', 0),      "harga",       "#4A148C", "#F3E5F5"),
        ('<i class="fa-solid fa-hashtag"></i>',  "MULTIPLIER", entity_counts.get('MULTIPLIER', 0), "jumlah/porsi","#BF360C", "#FBE9E7"),
    ]
    for col, (icon, lbl, val, sub, tc, bg) in zip(cols, ner_cards):
        with col:
            st.markdown(f"""
            <div class="ner-card" style="background:{bg};">
                <div class="ner-label" style="color:{tc};">{lbl}</div>
                <div class="ner-value" style="color:{tc};">{val:,}</div>
                <div style="font-size:20px;margin:6px 0;">{icon}</div>
                <div style="font-size:12px;color:#8AB89A;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    ents_per_rec = [len(x['entities']) for x in training_data]
    sorted_ents  = sorted(ents_per_rec)

    with col1:
        st.markdown("##### Statistik Dataset")
        stat_df = pd.DataFrame({"Metrik": [
            "Total records", "Total entitas", "Rata-rata entitas/record",
            "Median entitas/record", "Max entitas/record",
            "PERSON unik", "ITEM unik", "Synthetic templates"
        ], "Nilai": [
            f"{len(training_data):,}", f"{total_entities:,}",
            f"{sum(ents_per_rec)/len(ents_per_rec):.2f}",
            f"{sorted_ents[len(sorted_ents)//2]}",
            f"{max(ents_per_rec)}",
            f"{len(person_counter):,}", f"{len(item_counter):,}",
            f"{len(templates):,}"
        ]})
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

        st.markdown("##### Contoh Data Latih")
        samples = []
        for rec in training_data[:5]:
            entities_str = ", ".join(
                f"{e['label']}: {e['text']}"
                for e in rec['entities'][:4]
                if e['label'].strip() and not e['label'].startswith('[')
            )
            samples.append({"Kalimat": rec['text'][:80] + "...", "Entities": entities_str})
        st.dataframe(pd.DataFrame(samples), use_container_width=True, hide_index=True)

    with col2:
        ent_df = pd.DataFrame({
            'Entity': [k for k in entity_counts if k.strip() and not k.startswith('[')],
            'Count':  [v for k, v in entity_counts.items() if k.strip() and not k.startswith('[')]
        })
        fig = px.pie(ent_df, names='Entity', values='Count', hole=0.5,
                     color_discrete_map={
                         'PERSON': '#2E7D32', 'ITEM': '#1565C0',
                         'PRICE': '#6A1B9A', 'MULTIPLIER': '#BF360C'
                     })
        st.plotly_chart(plotly_layout(fig, 'Distribusi Entity Labels', height=300),
                        use_container_width=True)

        top_items = pd.DataFrame(item_counter.most_common(10), columns=['Item Menu', 'Frekuensi'])
        fig = px.bar(top_items.sort_values('Frekuensi'), x='Frekuensi', y='Item Menu',
                     orientation='h', color='Frekuensi', text_auto=True,
                     color_continuous_scale=['#A8D5B5', '#2D9A5A', '#0D4A30'], template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Top 10 Item Menu Terbanyak', height=320),
                        use_container_width=True)


# ── A/B TESTING ───────────────────────────────────────────
elif selected_section == "ab_testing":
    section_header("flask", "A/B Testing Simulation",
        "Simulasi perbandingan: Reminder Standar (A) vs Reminder Berbasis Rekomendasi Personal (B).")

    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Tentang Simulasi Ini</div>
        Data pengguna nyata Talang.in belum tersedia, sehingga simulasi menggunakan data sintetis.
        Tujuan: membuktikan potensi peningkatan respons pengguna jika reminder dibuat lebih personal.
        Metrik yang diukur: response rate, waktu penyelesaian tagihan, dan kepuasan pengguna.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: n_users   = st.slider("Jumlah Pengguna Simulasi", 100, 2000, 500, 100)
    with col2: base_rate  = st.slider("Base Response Rate (%)", 20, 60, 35, 5)
    with col3: effect     = st.slider("Effect Size Variant B (%)", 5, 30, 15, 5)

    np.random.seed(42)
    group_a = np.random.binomial(1, base_rate / 100, n_users // 2)
    group_b = np.random.binomial(1, (base_rate + effect) / 100, n_users // 2)
    rate_a, rate_b = group_a.mean() * 100, group_b.mean() * 100
    lift    = rate_b - rate_a
    time_a  = np.clip(np.random.normal(48, 12, n_users // 2), 1, None)
    time_b  = np.clip(np.random.normal(36, 10, n_users // 2), 1, None)
    sat_a   = np.clip(np.random.normal(3.2, 0.6, n_users // 2), 1, 5)
    sat_b   = np.clip(np.random.normal(3.8, 0.5, n_users // 2), 1, 5)

    col1, col2, col3 = st.columns(3)
    for col, metric, va, vb in [
        (col1, "Response Rate",       f"{rate_a:.1f}%",         f"{rate_b:.1f}%"),
        (col2, "Rata-rata Waktu",     f"{time_a.mean():.1f} jam", f"{time_b.mean():.1f} jam"),
        (col3, "Kepuasan (1-5)",      f"{sat_a.mean():.2f}",    f"{sat_b.mean():.2f}")
    ]:
        with col:
            st.markdown(f"""
            <div style="display:flex;gap:12px;">
                <div class="ab-result-card ab-variant-a" style="flex:1;">
                    <div class="ab-label" style="color:#1565C0;">Variant A</div>
                    <div style="font-size:11px;color:#6A9A7A;margin-bottom:8px;">Reminder Standar</div>
                    <div class="ab-stat" style="color:#1565C0;">{va}</div>
                    <div style="font-size:12px;color:#8AB89A;">{metric}</div>
                </div>
                <div class="ab-result-card ab-variant-b" style="flex:1;">
                    <div class="ab-label" style="color:#2E7D32;">Variant B</div>
                    <div style="font-size:11px;color:#6A9A7A;margin-bottom:8px;">Reminder Personal</div>
                    <div class="ab-stat" style="color:#2E7D32;">{vb}</div>
                    <div style="font-size:12px;color:#8AB89A;">{metric}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    winner = "Variant B (Reminder Personal) lebih unggul" if lift > 0 else "Variant A lebih unggul"
    st.markdown(f"""
    <div class="alert-box {'alert-success' if lift > 0 else 'alert-warning'}">
        <div class="alert-title"><i class="fa-solid fa-trophy"></i> Hasil Simulasi</div>
        Lift Response Rate: <span class="badge {'badge-green' if lift > 0 else 'badge-red'}">{lift:+.1f} ppt</span>
        &nbsp; Kesimpulan: <b>{winner}</b><br>
        Waktu penyelesaian lebih cepat <b>{time_a.mean() - time_b.mean():.1f} jam</b> dengan reminder personal.<br>
        <small style="color:#8AB89A;">Simulasi {n_users:,} pengguna sintetis - bukan data nyata Talang.in.</small>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            pd.DataFrame({'Variant': ['A - Standar', 'B - Personal'], 'Response Rate (%)': [rate_a, rate_b]}),
            x='Variant', y='Response Rate (%)', color='Variant', text_auto='.1f',
            color_discrete_map={'A - Standar': '#1565C0', 'B - Personal': '#2E7D32'},
            template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Perbandingan Response Rate', height=350),
                        use_container_width=True)
    with col2:
        time_df = pd.DataFrame({
            'Waktu (jam)': list(time_a[:200]) + list(time_b[:200]),
            'Variant': ['A - Standar'] * 200 + ['B - Personal'] * 200
        })
        fig = px.histogram(time_df, x='Waktu (jam)', color='Variant', nbins=30,
                           barmode='overlay', opacity=0.7,
                           color_discrete_map={'A - Standar': '#1565C0', 'B - Personal': '#2E7D32'},
                           template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Distribusi Waktu Penyelesaian Tagihan', height=350),
                        use_container_width=True)


# ── KESIMPULAN ────────────────────────────────────────────
elif selected_section == "kesimpulan":
    section_header("clipboard-check", "Kesimpulan",
        "Rangkuman akhir proses Data Science Talang.in dari awal hingga dataset final siap pakai.")

    st.markdown("""
    <div class="alert-box alert-success">
        <div class="alert-title"><i class="fa-solid fa-clipboard-check"></i> Kesimpulan Project Data Science Talang.in</div>
    </div>
    """, unsafe_allow_html=True)

    kesimpulan_items = [
        ("file-import",     "1. Pengumpulan Data",
         f"6 dataset dikumpulkan dari sumber publik. Data utama adalah <b>gofood_dataset.csv</b> ({len(df_raw):,} baris) sebagai sumber variasi entity ITEM dan PRICE, serta <b>indonesian-names.csv</b> untuk entity PERSON."),
        ("broom",           "2. Data Cleaning",
         f"Dataset GoFood dibersihkan dari <b>{duplicate_before:,} duplikat</b> dan <b>{missing_before:,} missing value</b>, menghasilkan <b>{len(df):,} data bersih</b> yang siap dianalisis."),
        ("chart-bar",       "3. EDA",
         f"Eksplorasi menunjukkan keragaman <b>{total_area} area</b>, <b>{total_merchant:,} merchant</b>, dan berbagai kategori dengan rata-rata harga <b>Rp {avg_price:,.0f}</b> - semua mendukung variasi data latih NER."),
        ("robot",           "4. Dataset Final NER",
         f"<b>{len(training_data):,} kalimat transaksi</b> berlabel entity (<b>{total_entities:,} entitas</b> total) dihasilkan menggunakan template + LLM Gemma. File <b>training_data.json</b> siap diserahkan ke tim AI Engineer."),
        ("flask",           "5. A/B Testing",
         "Simulasi menunjukkan potensi peningkatan response rate dengan reminder berbasis rekomendasi personal dibanding reminder standar."),
        ("heart-pulse",     "6. Health Score Project",
         f"Health Score: <b style='color:{health_color};'>{health_score}% - {health_status}</b>. Data bersih, pipeline terdokumentasi, dan dataset final siap untuk fitur <b>AI Smart Transaction Input</b>."),
    ]

    for icon, title, body in kesimpulan_items:
        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-{icon}"></i> {title}</div>
            {body}
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    checks_col1 = [
        ("Bebas Duplikat",            True,  f"{duplicate_before:,} duplikat dihapus"),
        ("Missing Value Tertangani",  True,  f"fillna(0) pada discount_price"),
        ("Data Cleaning Terdokumentasi", True, "sebelum & sesudah cleaning"),
    ]
    checks_col2 = [
        ("EDA Terdokumentasi",        True,  "4 aspek: kategori, merchant, harga, area"),
        ("Insight Terhubung ke NER",  True,  "setiap insight dikaitkan ke kebutuhan model"),
        ("A/B Testing Diimplementasi", True, "simulasi reminder standar vs personal"),
    ]
    checks_col3 = [
        ("Dataset Final Tersedia",    True,  f"{len(training_data):,} records NER"),
        ("4 Entity Teranotasi",       True,  "PERSON, ITEM, PRICE, MULTIPLIER"),
        ("Pipeline Terdokumentasi",   True,  "2 notebook: preprocessing + data gen"),
    ]

    for col, checks in [(col1, checks_col1), (col2, checks_col2), (col3, checks_col3)]:
        with col:
            for label, passed, detail in checks:
                icon_class = "fa-circle-check" if passed else "fa-circle-xmark"
                icon_color = "#2E7D32" if passed else "#C62828"
                badge_class = "badge-green" if passed else "badge-red"
                badge_text = "Lulus" if passed else "Perlu Perbaikan"
                html = f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding:8px 0;border-bottom:1px solid #E8F5E9;"><span style="font-size:13px;color:#2D5A3D;"><i class="fa-solid {icon_class}" style="color:{icon_color};"></i> {label}<br><small style="color:#8AB89A;">{detail}</small></span><span class="badge {badge_class}">{badge_text}</span></div>'
                st.markdown(html, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-area">
    <div class="footer-title"><i class="fa-solid fa-wallet"></i> Talang.in - Data Science Dashboard</div>
    <p>NER | GoFood | Analytics | Streamlit Cloud</p>
    <p style="margin-top:8px;">
        <span class="badge badge-green">v4.0</span>&nbsp;
        <span class="badge badge-green">Streamlit Cloud</span>&nbsp;
        <span class="badge badge-blue">{len(training_data):,} Data Latih</span>&nbsp;
        <span class="badge badge-blue">{total_entities:,} Entities</span>
    </p>
    <p style="margin-top:12px;font-size:12px;color:#8AB89A;">
        Data Utama: gofood_dataset | indonesian-names
        <i class="fa-solid fa-circle" style="font-size:4px;vertical-align:middle;margin:0 6px;"></i>
        Data Pendukung: indonesian_food | steakhouse
        <i class="fa-solid fa-circle" style="font-size:4px;vertical-align:middle;margin:0 6px;"></i>
        Data Eksplorasi: alergen | nutrition
    </p>
</div>
""", unsafe_allow_html=True)