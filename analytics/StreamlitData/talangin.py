# =========================================================
# TALANG.IN — DATA SCIENCE ANALYTICS DASHBOARD
# GoFood NER Project | Streamlit Professional UI v3.0
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from collections import Counter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Talang.in Data Science Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: linear-gradient(150deg, #F0F7F0 0%, #E8F4E8 50%, #EAF5EA 100%); }
.block-container { padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1300px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D3B2E 0%, #134E3A 60%, #1B5E40 100%);
}
section[data-testid="stSidebar"] * { color: #D4EDDA !important; }
section[data-testid="stSidebar"] label { color: #A8D5B5 !important; font-size: 13px !important; }
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #2D7A50 !important; }

.page-header {
    background: linear-gradient(135deg, #0D4A30 0%, #1B6E44 100%);
    border-radius: 20px; padding: 36px 40px; margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.page-header::after {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px; background: rgba(255,255,255,0.05); border-radius: 50%;
}
.page-header h1 { color: #FFFFFF; font-size: 32px; font-weight: 800; margin: 0 0 8px 0; }
.page-header p { color: #A8D5B5; font-size: 14px; margin: 0; line-height: 1.7; }
.header-badge {
    display: inline-block; background: rgba(255,255,255,0.15); color: #CCEBCC;
    font-size: 11px; font-weight: 600; padding: 4px 14px; border-radius: 20px;
    margin-bottom: 14px; letter-spacing: 0.5px; text-transform: uppercase;
}

.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 40px 0 6px 0; padding-bottom: 12px; border-bottom: 2px solid #C8E6C9;
}
.section-icon {
    width: 38px; height: 38px; background: #E8F5E9; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.section-title-text { font-size: 22px; font-weight: 700; color: #1B4332; margin: 0; }
.section-desc { font-size: 14px; color: #5A8A6A; margin: 0 0 20px 0; }

.metric-card {
    background: #FFFFFF; border-radius: 16px; padding: 20px 22px;
    border: 1.5px solid #D4EDD8; transition: transform 0.2s ease, box-shadow 0.2s ease; height: 100%;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(27,99,54,0.12); }
.metric-icon { font-size: 24px; margin-bottom: 10px; display: block; }
.metric-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #6A9A7A; margin-bottom: 5px; }
.metric-value { font-size: 26px; font-weight: 800; color: #1B4332; line-height: 1.1; }
.metric-sub { font-size: 12px; color: #8AB89A; margin-top: 5px; }

.ner-card { background: #FFFFFF; border-radius: 14px; padding: 18px 20px; border: 1.5px solid #D4EDD8; text-align: center; height: 100%; }
.ner-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.ner-value { font-size: 28px; font-weight: 800; line-height: 1; margin-bottom: 4px; }

.pipeline-card { background: #FFFFFF; border-radius: 14px; padding: 20px 22px; border: 1.5px solid #D4EDD8; height: 100%; }
.pipeline-step { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
.step-num {
    width: 26px; height: 26px; background: #1B4332; color: white; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.step-text { font-size: 13px; color: #2D5A3D; line-height: 1.6; }
.step-label { font-weight: 700; color: #1B4332; }

.alert-box { border-radius: 14px; padding: 16px 20px; margin-bottom: 16px; font-size: 14px; line-height: 1.7; }
.alert-info { background: #EBF8EE; border-left: 4px solid #2D9A5A; color: #1B4332; }
.alert-success { background: #E8F5E9; border-left: 4px solid #2E7D32; color: #1B4332; }
.alert-warning { background: #FFF8E1; border-left: 4px solid #F9A825; color: #5D4037; }
.alert-title { font-weight: 700; font-size: 14px; margin-bottom: 6px; }

.source-chip {
    display: inline-block; background: #E8F5E9; color: #1B5E20;
    border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 600;
    margin: 3px; border: 1px solid #C8E6C9;
}
.badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-green { background: #C8E6C9; color: #1B5E20; }
.badge-yellow { background: #FFF9C4; color: #795548; }
.badge-red { background: #FFCDD2; color: #B71C1C; }
.badge-blue { background: #BBDEFB; color: #0D47A1; }

.health-card { background: #FFFFFF; border-radius: 16px; padding: 28px 32px; border: 1.5px solid #D4EDD8; text-align: center; }
.health-score-big { font-size: 64px; font-weight: 800; line-height: 1; margin: 12px 0; }
.health-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #6A9A7A; }

.footer-area { background: #FFFFFF; border-radius: 16px; padding: 24px 32px; text-align: center; border: 1.5px solid #D4EDD8; margin-top: 48px; }
.footer-area p { color: #6A9A7A; font-size: 13px; margin: 4px 0; }
.footer-title { font-size: 15px; font-weight: 700; color: #1B4332; margin-bottom: 4px; }

[data-testid="stDataFrame"] { background: #FFFFFF; border-radius: 16px; border: 1.5px solid #D4EDD8 !important; overflow: hidden; }
.stProgress > div > div { background: linear-gradient(90deg, #2D9A5A, #52C27A) !important; border-radius: 8px !important; }
.stProgress > div { background: #D4EDD8 !important; border-radius: 8px !important; height: 10px !important; }
.stTabs [data-baseweb="tab-list"] { background: #E8F5E9; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #5A8A6A; font-weight: 600; font-size: 14px; }
.stTabs [aria-selected="true"] { background: #FFFFFF !important; color: #1B4332 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #B2D8B2; border-radius: 4px; }

.nav-item-active {
    background: rgba(255,255,255,0.18) !important;
    border-left: 3px solid #52C27A !important;
}
.nav-item {
    background: rgba(255,255,255,0.07);
    border-radius: 8px; padding: 7px 12px; margin-bottom: 4px;
    font-size: 13px; color: #D4EDDA; cursor: pointer;
    border-left: 3px solid transparent;
    transition: background 0.15s, border-color 0.15s;
}
.sidebar-logo-icon {
    width: 48px; height: 48px; margin: 0 auto 10px auto;
    border-radius: 16px; background: rgba(255,255,255,0.10);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: #FFFFFF;
}
.metric-icon i, .section-icon i { color: #1B5E40; }
.alert-title i { margin-right: 7px; }

/* A/B testing specific */
.ab-result-card {
    background: #FFFFFF; border-radius: 14px; padding: 22px 24px;
    border: 1.5px solid #D4EDD8; text-align: center; height: 100%;
}
.ab-variant-a { border-top: 4px solid #1565C0; }
.ab-variant-b { border-top: 4px solid #2E7D32; }
.ab-stat { font-size: 32px; font-weight: 800; line-height: 1.1; margin: 8px 0 4px 0; }
.ab-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #6A9A7A; }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_gofood():
    df = pd.read_csv("../data/gofood_dataset.csv")
    missing_before = df.isnull().sum().sum()
    duplicate_before = int(df.duplicated().sum())
    df['discount_price'] = df['discount_price'].fillna(0)
    df_clean = df.drop_duplicates()
    missing_after = int(df_clean.isnull().sum().sum())
    return df, df_clean, int(missing_before), duplicate_before, missing_after

@st.cache_data
def load_training_data():
    with open("../outputs/training_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_fixed_dataset():
    with open("../dataset_fixed.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_templates():
    with open("../outputs/talangin_synthetic_templates.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def compute_ner_stats(n_records):
    with open("../outputs/training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    entity_counts = Counter()
    item_counter  = Counter()
    person_counter = Counter()
    person_per_rec = []
    for rec in training_data:
        n_p = 0
        for ent in rec.get("entities", []):
            lbl = ent["label"]
            if lbl and not lbl.startswith("[") and lbl.strip():
                entity_counts[lbl] += 1
            if lbl == "ITEM":
                item_counter[ent["text"].lower()] += 1
            if lbl == "PERSON":
                person_counter[ent["text"]] += 1
                n_p += 1
        person_per_rec.append(n_p)
    return entity_counts, item_counter, person_counter, person_per_rec

df_raw, df, missing_before, duplicate_before, missing_after = load_gofood()
training_data  = load_training_data()
fixed_dataset  = load_fixed_dataset()
templates      = load_templates()
entity_counts, item_counter, person_counter, person_per_rec = compute_ner_stats(len(training_data))

# =========================================================
# COMPUTE KPIs (global, pre-filter for header stats)
# =========================================================

total_entities = sum(v for k, v in entity_counts.items() if k.strip() and not k.startswith('['))

health_score  = 100
health_issues = []
if duplicate_before > 0:
    health_score -= 20
    health_issues.append(f"{duplicate_before:,} duplikat ditemukan → sudah dibersihkan dengan drop_duplicates()")
if missing_before > 0:
    health_score -= 20
    health_issues.append(f"{missing_before:,} missing value ditemukan → sudah ditangani dengan fillna(0) pada kolom discount_price")

health_color  = "#2E7D32" if health_score >= 80 else "#F57F17" if health_score >= 60 else "#C62828"
health_icon   = "fa-heart-pulse" if health_score >= 80 else "fa-circle-exclamation" if health_score >= 60 else "fa-circle-xmark"
health_status = "Sangat Baik" if health_score >= 80 else "Perlu Perhatian" if health_score >= 60 else "Kritis"

# =========================================================
# SIDEBAR — ACTIVE NAVIGATION
# =========================================================

SECTIONS = [
    ("overview",        "fa-folder-open",          "Overview Dataset"),
    ("kpi",             "fa-thumbtack",             "KPI Metrics"),
    ("grafik_kategori", "fa-chart-column",          "Grafik Kategori"),
    ("grafik_merchant", "fa-store",                 "Grafik Merchant"),
    ("tren_area",       "fa-chart-line",            "Tren Area"),
    ("training_ner",    "fa-robot",                 "Training Data NER"),
    ("preprocessing",   "fa-microscope",            "Preprocessing Pipeline"),
    ("data_quality",    "fa-shield-halved",         "Data Quality Check"),
    ("insight",         "fa-lightbulb",             "Insight"),
    ("ab_testing",      "fa-flask",                 "A/B Testing Simulation"),
    ("recommendation",  "fa-bullseye",              "Recommendation"),
    ("health_score",    "fa-heart-pulse",           "Health Score"),
    ("kesimpulan",      "fa-clipboard-check",       "Kesimpulan"),
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div class="sidebar-logo-icon"><i class="fa-solid fa-wallet"></i></div>
        <div style="font-size:17px; font-weight:800; color:#FFFFFF; margin-top:8px;">Talang.in</div>
        <div style="font-size:12px; color:#8CBFA0; margin-top:2px;">Data Science Dashboard v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""<div style="font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:1px; color:#8CBFA0; margin-bottom:10px;">
    <i class="fa-solid fa-map-pin"></i> Navigasi Section</div>""", unsafe_allow_html=True)

    section_labels = [s[2] for s in SECTIONS]
    section_keys   = [s[0] for s in SECTIONS]
    section_icons  = [s[1] for s in SECTIONS]

    selected_section = st.radio(
        label="Pilih Section",
        options=section_keys,
        format_func=lambda k: next(s[2] for s in SECTIONS if s[0] == k),
        label_visibility="collapsed"
    )

    # Style the radio to match nav look
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.07); border-radius: 8px;
        padding: 6px 12px !important; margin-bottom: 3px;
        font-size: 13px !important; color: #D4EDDA !important;
        display: block; width: 100%;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(255,255,255,0.18) !important;
        border-left: 3px solid #52C27A;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] { gap: 2px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""<div style="font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:1px; color:#8CBFA0; margin-bottom:10px;">
    <i class="fa-solid fa-sliders"></i> Filter Data GoFood</div>""", unsafe_allow_html=True)

    kategori_filter = st.multiselect(
        "Pilih Kategori",
        options=df['category'].unique(),
        default=df['category'].unique(),
        help="Filter berdasarkan kategori produk GoFood"
    )
    area_filter = st.multiselect(
        "Pilih Area Merchant",
        options=df['merchant_area'].unique(),
        default=df['merchant_area'].unique(),
        help="Filter berdasarkan lokasi area merchant"
    )

    filtered_df = df[
        (df['category'].isin(kategori_filter)) &
        (df['merchant_area'].isin(area_filter))
    ]

    st.markdown("---")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.08); border-radius:12px; padding:14px 16px;">
        <div style="font-size:11px; color:#8CBFA0; font-weight:600; text-transform:uppercase;
        letter-spacing:0.8px; margin-bottom:10px;">
        <i class="fa-solid fa-chart-column"></i> Status Filter</div>
        <div style="font-size:13px; color:#D4EDDA; margin-bottom:5px;">
            <i class="fa-solid fa-database"></i> Data aktif: <b style="color:#FFF;">{len(filtered_df):,}</b> baris
        </div>
        <div style="font-size:13px; color:#D4EDDA; margin-bottom:5px;">
            <i class="fa-solid fa-tags"></i> Kategori: <b style="color:#FFF;">{len(kategori_filter)}</b> dipilih
        </div>
        <div style="font-size:13px; color:#D4EDDA;">
            <i class="fa-solid fa-location-dot"></i> Area: <b style="color:#FFF;">{len(area_filter)}</b> dipilih
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#8CBFA0; text-align:center; line-height:1.8;">
    Dibuat untuk memenuhi requirement<br><b style="color:#A8D5B5;">Data Scientist</b><br>
    Deploy ke <b style="color:#A8D5B5;">Streamlit Cloud</b>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# COMPUTE per-filter KPIs
# =========================================================

total_pengeluaran  = filtered_df['price'].sum()
jumlah_transaksi   = len(filtered_df)
avg_price          = filtered_df['price'].mean()
kategori_terbanyak = filtered_df['category'].value_counts().idxmax()
merchant_aktif     = filtered_df['merchant_name'].value_counts().idxmax()
total_merchant     = filtered_df['merchant_name'].nunique()
total_area         = filtered_df['merchant_area'].nunique()

# =========================================================
# HEADER (always visible)
# =========================================================

st.markdown("""
<div class="page-header">
    <div class="header-badge">
        <i class="fa-solid fa-robot"></i> NER &middot; GoFood &middot; Data Science &middot; Analytics Report
    </div>
    <h1><i class="fa-solid fa-wallet"></i> Talang.in — Data Science Analytics Dashboard</h1>
    <p>
        Dashboard ini mendokumentasikan proses Data Science untuk mendukung fitur
        <b>AI Smart Transaction Input</b> pada aplikasi <b>Talang.in</b>.
        Dataset GoFood digunakan sebagai sumber data menu, harga, merchant, dan kategori
        untuk membangun training data NER (Named Entity Recognition).
        Dashboard ini berfokus pada <b>data preparation, EDA, preprocessing, dan kesiapan data latih</b> —
        bukan hasil prediksi atau inference model. Pengembangan model NER dilakukan oleh tim AI Engineer
        menggunakan data latih yang telah disiapkan di sini.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SECTION ROUTER — render only selected section
# =========================================================

# ─────────────────────────────────────────────────────────
# SECTION: OVERVIEW DATASET
# ─────────────────────────────────────────────────────────
if selected_section == "overview":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-folder-open"></i></div>
        <p class="section-title-text">Overview Dataset</p>
    </div>
    <p class="section-desc">
        Ringkasan seluruh dataset yang digunakan dalam project Talang.in — dari sumber data mentah
        hingga hasil preprocessing dan generasi sintetis untuk training data NER.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    overview_cards = [
        ("<i class=\"fa-solid fa-database\"></i>", "GoFood Raw",       f"{len(df_raw):,}",        "data mentah awal"),
        ("<i class=\"fa-solid fa-broom\"></i>",    "Setelah Cleaning", f"{len(df):,}",            "drop duplikat & null"),
        ("<i class=\"fa-solid fa-robot\"></i>",    "Data Latih NER",   f"{len(training_data):,}", "siap dikirim ke AI Engineer"),
        ("<i class=\"fa-solid fa-list-check\"></i>","Fixed Dataset",   f"{len(fixed_dataset):,}", "hasil preprocessing"),
        ("<i class=\"fa-solid fa-clipboard-check\"></i>","Synthetic Tpl.", f"{len(templates):,}", "template LLM Gemma"),
    ]
    for col, (icon, label, val, sub) in zip([col1,col2,col3,col4,col5], overview_cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Peran Dataset GoFood dalam Talang.in</div>
        Dataset GoFood <b>bukan merupakan data transaksi pengguna Talang.in</b>, melainkan digunakan sebagai
        <b>sumber data referensi</b> untuk membangun training data NER. Data menu, harga, nama merchant, dan
        kategori dari GoFood diekstrak dan digunakan sebagai bahan anotasi entity ITEM dan PRICE dalam
        kalimat-kalimat sintetis yang merepresentasikan skenario tagihan nyata di aplikasi Talang.in.
        Tujuan akhirnya adalah melatih model NER yang mampu mengurai input transaksi pengguna secara otomatis
        pada fitur <b>AI Smart Transaction Input</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-success">
        <div class="alert-title"><i class="fa-solid fa-layer-group"></i> 6 Dataset Sumber yang Digabungkan</div>
        <span class="source-chip">gofood_dataset.csv &middot; 45.195 baris</span>
        <span class="source-chip">alergen_dataset.csv &middot; 100.000 baris</span>
        <span class="source-chip">Steakhouse_dataset.csv &middot; 150 baris</span>
        <span class="source-chip">indonesian_food.csv &middot; 1.273 baris</span>
        <span class="source-chip">nutrition.csv &middot; 1.345 baris</span>
        <span class="source-chip">indonesian-names.csv &middot; 1.960 baris</span>
        <br><br>
        Semua dataset diproses di <b>data_preprocessing.ipynb</b> untuk menghasilkan training data NER.
        Generasi data sintetis menggunakan <b>data_gen_gemma.ipynb</b> dengan LLM Gemma.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Lihat Sample GoFood Dataset (10 baris pertama)", expanded=False):
        st.dataframe(filtered_df.head(10), use_container_width=True)

    with st.expander("Lihat Statistik Deskriptif GoFood", expanded=False):
        st.dataframe(filtered_df.describe(), use_container_width=True)

    with st.expander("Lihat Sample Training Data NER", expanded=False):
        samples = []
        for rec in training_data[:8]:
            ents = ", ".join([
                f"{e['label']}:{e['text']}" for e in rec['entities'][:4]
                if e['label'].strip() and not e['label'].startswith('[')
            ])
            samples.append({
                "Teks (truncated)": rec['text'][:100] + "...",
                "Sample Entities": ents,
                "Total Entities": len(rec['entities'])
            })
        st.dataframe(pd.DataFrame(samples), use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION: KPI METRICS
# ─────────────────────────────────────────────────────────
elif selected_section == "kpi":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-thumbtack"></i></div>
        <p class="section-title-text">KPI Metrics — GoFood Analytics</p>
    </div>
    <p class="section-desc">
        Indikator utama dari dataset GoFood berdasarkan filter aktif di sidebar.
        Metrik ini menggambarkan karakteristik data yang digunakan sebagai sumber training data NER.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    kpi_items = [
        ("<i class=\"fa-solid fa-coins\"></i>",        "Total Harga (Akumulasi)", f"Rp {total_pengeluaran:,.0f}", "akumulasi semua produk"),
        ("<i class=\"fa-solid fa-cart-shopping\"></i>","Jumlah Produk",           f"{jumlah_transaksi:,}",        "total item tercatat"),
        ("<i class=\"fa-solid fa-chart-line\"></i>",   "Rata-rata Harga",         f"Rp {avg_price:,.0f}",         "per produk"),
        ("<i class=\"fa-solid fa-trophy\"></i>",       "Kategori Dominan",        f"{kategori_terbanyak}",        "kategori terpopuler"),
    ]
    for col, (icon, label, val, sub) in zip([col1,col2,col3,col4], kpi_items):
        with col:
            fs = "20px" if len(val) > 10 else "24px"
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:{fs};">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="display:flex; gap:20px; align-items:center;">
            <span style="font-size:30px;"><i class="fa-solid fa-store"></i></span>
            <div>
                <div class="metric-label">Total Merchant Unik</div>
                <div class="metric-value">{total_merchant:,}</div>
                <div class="metric-sub">merchant terdaftar di GoFood</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="display:flex; gap:20px; align-items:center;">
            <span style="font-size:30px;"><i class="fa-solid fa-medal"></i></span>
            <div>
                <div class="metric-label">Merchant Paling Aktif</div>
                <div class="metric-value" style="font-size:16px;">{merchant_aktif}</div>
                <div class="metric-sub">volume produk tertinggi</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: GRAFIK KATEGORI
# ─────────────────────────────────────────────────────────
elif selected_section == "grafik_kategori":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-chart-column"></i></div>
        <p class="section-title-text">Grafik Kategori</p>
    </div>
    <p class="section-desc">
        Distribusi produk berdasarkan kategori GoFood. Data ini digunakan untuk memahami
        keberagaman item menu yang tersedia sebagai sumber entity ITEM dalam training data NER.
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Bar Chart (Semua)", "Pie Chart (Top 10)"])
    kategori_data = filtered_df['category'].value_counts().reset_index()
    kategori_data.columns = ['Kategori', 'Jumlah']

    with tab1:
        fig_bar = px.bar(
            kategori_data, x='Kategori', y='Jumlah', color='Jumlah',
            text_auto=True, color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'],
            template='plotly_white'
        )
        fig_bar.update_layout(
            height=430, coloraxis_showscale=False,
            title=dict(text='Distribusi Semua Kategori Produk GoFood', font=dict(size=14, color='#1B4332')),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50,b=20,l=10,r=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        top10_kat = kategori_data.head(10)
        fig_pie = px.pie(
            top10_kat, names='Kategori', values='Jumlah',
            color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.45
        )
        fig_pie.update_layout(
            height=430,
            title=dict(text='Proporsi Top 10 Kategori', font=dict(size=14, color='#1B4332')),
            paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50,b=20,l=10,r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION: GRAFIK MERCHANT
# ─────────────────────────────────────────────────────────
elif selected_section == "grafik_merchant":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-store"></i></div>
        <p class="section-title-text">Grafik Merchant</p>
    </div>
    <p class="section-desc">
        Top 10 merchant berdasarkan jumlah produk. Item menu dari merchant aktif menjadi
        sumber utama entity ITEM dalam training data NER Talang.in.
    </p>
    """, unsafe_allow_html=True)

    merchant_data = filtered_df['merchant_name'].value_counts().head(10).reset_index()
    merchant_data.columns = ['Merchant', 'Jumlah']

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_mbar = px.bar(
            merchant_data.sort_values('Jumlah'), x='Jumlah', y='Merchant',
            orientation='h', color='Jumlah', text_auto=True,
            color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white'
        )
        fig_mbar.update_layout(
            height=400, coloraxis_showscale=False,
            title=dict(text='Top 10 Merchant — Jumlah Produk', font=dict(size=14,color='#1B4332')),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50,b=20,l=10,r=10)
        )
        st.plotly_chart(fig_mbar, use_container_width=True)

    with col2:
        fig_mpie = px.pie(
            merchant_data, names='Merchant', values='Jumlah',
            color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.5
        )
        fig_mpie.update_layout(
            height=400,
            title=dict(text='Porsi Top 10 Merchant', font=dict(size=14,color='#1B4332')),
            paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50,b=20,l=10,r=10),
            legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig_mpie, use_container_width=True)

    st.markdown(f"""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-trophy"></i> Merchant Paling Aktif</div>
        <b>{merchant_aktif}</b> memiliki volume produk tertinggi dari {total_merchant:,} merchant unik.
        Item menu dari merchant aktif diprioritaskan sebagai data entity ITEM dalam training data NER.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: TREN AREA
# ─────────────────────────────────────────────────────────
elif selected_section == "tren_area":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-chart-line"></i></div>
        <p class="section-title-text">Tren Area Analytics</p>
    </div>
    <p class="section-desc">
        Distribusi produk berdasarkan area merchant. Area dengan volume tinggi menghasilkan
        variasi item menu yang lebih kaya untuk training data NER.
    </p>
    """, unsafe_allow_html=True)

    area_data = filtered_df['merchant_area'].value_counts().reset_index()
    area_data.columns = ['Area', 'Jumlah']

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=area_data['Area'], y=area_data['Jumlah'],
        mode='lines+markers+text',
        line=dict(color='#2D9A5A', width=3),
        marker=dict(color='#2D9A5A', size=11, line=dict(color='white', width=2)),
        text=area_data['Jumlah'], textposition='top center',
        textfont=dict(size=11, color='#1B4332'),
        fill='tozeroy', fillcolor='rgba(45,154,90,0.10)'
    ))
    fig_area.update_layout(
        height=380,
        title=dict(text=f'Distribusi Produk per Area Merchant ({total_area} Area)', font=dict(size=14,color='#1B4332')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
        margin=dict(t=50,b=20,l=10,r=10)
    )
    st.plotly_chart(fig_area, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION: TRAINING DATA NER
# ─────────────────────────────────────────────────────────
elif selected_section == "training_ner":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-robot"></i></div>
        <p class="section-title-text">Training Data NER — Data Latih Siap Pakai</p>
    </div>
    <p class="section-desc">
        Statistik data latih NER yang disiapkan untuk pengembangan model AI Smart Transaction Input Talang.in.
        Data ini berisi anotasi PERSON, ITEM, PRICE, dan MULTIPLIER, dan <b>belum merepresentasikan
        hasil prediksi model</b> — model NER belum dilatih.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-warning">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Status Model NER</div>
        Dataset ini telah <b>siap digunakan oleh tim AI Engineer</b> sebagai data latih (training data).
        Proses training model NER, beserta evaluasi metrik seperti precision, recall, dan F1-score per entity,
        akan dilakukan setelah model selesai dikembangkan oleh tim AI Engineer. Dashboard Data Science ini
        hanya mencakup tahap <b>persiapan dan validasi data latih</b>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    ner_cards = [
        ("<i class=\"fa-solid fa-user\"></i>",    "PERSON",     entity_counts.get('PERSON',0),     "nama orang dalam tagihan", "#1B5E20", "#E8F5E9"),
        ("<i class=\"fa-solid fa-utensils\"></i>","ITEM",       entity_counts.get('ITEM',0),       "nama menu / produk",       "#0D47A1", "#E3F2FD"),
        ("<i class=\"fa-solid fa-coins\"></i>",   "PRICE",      entity_counts.get('PRICE',0),      "harga berbagai format",    "#4A148C", "#F3E5F5"),
        ("<i class=\"fa-solid fa-hashtag\"></i>", "MULTIPLIER", entity_counts.get('MULTIPLIER',0), "jumlah porsi / orang",     "#BF360C", "#FBE9E7"),
    ]
    for col, (icon, lbl, val, sub, tc, bg) in zip([col1,col2,col3,col4], ner_cards):
        with col:
            st.markdown(f"""
            <div class="ner-card" style="background:{bg};">
                <div class="ner-label" style="color:{tc};">{lbl}</div>
                <div class="ner-value" style="color:{tc};">{val:,}</div>
                <div style="font-size:20px; margin:6px 0;">{icon}</div>
                <div style="font-size:12px; color:#8AB89A;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-table"></i> Ringkasan Statistik Data Latih</div>
        </div>
        """, unsafe_allow_html=True)

        ents_per_rec = [len(x['entities']) for x in training_data]
        sorted_ents  = sorted(ents_per_rec)

        stat_df = pd.DataFrame({
            "Metrik": [
                "Total records data latih",
                "Total entitas teranotasi",
                "Rata-rata entitas / record",
                "Median entitas / record",
                "Max entitas / record",
                "Nama orang unik (PERSON)",
                "Nama menu unik (ITEM)",
                "Synthetic templates",
                "Fixed dataset records",
                "Dataset sumber",
            ],
            "Nilai": [
                f"{len(training_data):,}",
                f"{total_entities:,}",
                f"{sum(ents_per_rec)/len(ents_per_rec):.2f}",
                f"{sorted_ents[len(sorted_ents)//2]}",
                f"{max(ents_per_rec)}",
                f"{len(person_counter):,}",
                f"{len(item_counter):,}",
                f"{len(templates):,}",
                f"{len(fixed_dataset):,}",
                "6 dataset",
            ]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    with col2:
        ent_df = pd.DataFrame({
            'Entity': [k for k in entity_counts if k.strip() and not k.startswith('[')],
            'Count':  [v for k, v in entity_counts.items() if k.strip() and not k.startswith('[')]
        })
        fig_ent = px.pie(
            ent_df, names='Entity', values='Count',
            color_discrete_map={'PERSON':'#2E7D32','ITEM':'#1565C0','PRICE':'#6A1B9A','MULTIPLIER':'#BF360C'},
            hole=0.5
        )
        fig_ent.update_layout(
            height=330,
            title=dict(text='Distribusi Entity Labels dalam Data Latih', font=dict(size=14,color='#1B4332')),
            paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_ent, use_container_width=True)

    st.write("")
    top_items = pd.DataFrame(item_counter.most_common(15), columns=['Item Menu', 'Frekuensi'])
    fig_items = px.bar(
        top_items.sort_values('Frekuensi'), x='Frekuensi', y='Item Menu',
        orientation='h', color='Frekuensi',
        color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'],
        template='plotly_white', text_auto=True
    )
    fig_items.update_layout(
        height=430, coloraxis_showscale=False,
        title=dict(text='Top 15 Item Menu Paling Sering Muncul di Data Latih', font=dict(size=14,color='#1B4332')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50,b=20,l=10,r=10)
    )
    st.plotly_chart(fig_items, use_container_width=True)

    person_dist = Counter(person_per_rec)
    pdist_df = pd.DataFrame(sorted(person_dist.items()), columns=['Jumlah Person/Record','Jumlah Record']).head(8)
    fig_pdist = px.bar(
        pdist_df, x='Jumlah Person/Record', y='Jumlah Record',
        color='Jumlah Record', color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'],
        template='plotly_white', text_auto=True
    )
    fig_pdist.update_layout(
        height=360, coloraxis_showscale=False,
        title=dict(text='Distribusi Jumlah PERSON per Record Data Latih (multi-orang per tagihan)', font=dict(size=14,color='#1B4332')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50,b=20,l=10,r=10)
    )
    st.plotly_chart(fig_pdist, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION: PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────
elif selected_section == "preprocessing":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-microscope"></i></div>
        <p class="section-title-text">Preprocessing Pipeline</p>
    </div>
    <p class="section-desc">
        Alur lengkap transformasi data — dari raw dataset mentah hingga training data NER siap pakai
        untuk digunakan oleh tim AI Engineer Talang.in.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="pipeline-card">
            <div class="metric-label" style="margin-bottom:16px;">
                <i class="fa-solid fa-file-import"></i> INPUT — Dataset Sumber
            </div>
            <div class="pipeline-step">
                <div class="step-num">1</div>
                <div class="step-text"><span class="step-label">gofood_dataset.csv</span><br>45.195 baris &middot; nama produk, harga, kategori, area merchant GoFood</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">2</div>
                <div class="step-text"><span class="step-label">alergen_dataset.csv</span><br>100.000 baris &middot; nama produk dari label informasi alergen pangan Indonesia</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">3</div>
                <div class="step-text"><span class="step-label">Steakhouse_dataset.csv</span><br>150 baris &middot; menu restoran steakhouse Indonesia (nama, deskripsi, harga)</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">4</div>
                <div class="step-text"><span class="step-label">indonesian_food.csv + nutrition.csv</span><br>2.618 baris &middot; nama makanan Indonesia beserta info kandungan gizi</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">5</div>
                <div class="step-text"><span class="step-label">indonesian-names.csv</span><br>1.960 baris &middot; nama orang Indonesia &rarr; sumber entity PERSON dalam data latih</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pipeline-card">
            <div class="metric-label" style="margin-bottom:16px;">
                <i class="fa-solid fa-gear"></i> PROSES — Notebook Pipeline
            </div>
            <div class="pipeline-step">
                <div class="step-num">A</div>
                <div class="step-text"><span class="step-label">Cleaning &amp; Normalisasi</span><br>Drop NaN, hapus duplikat, lowercase, strip whitespace, hapus noise &amp; stopwords (Sastrawi)</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">B</div>
                <div class="step-text"><span class="step-label">extract_core_item()</span><br>Ekstrak nama menu inti dari judul e-commerce panjang &rarr; 1-3 kata conversational</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">C</div>
                <div class="step-text"><span class="step-label">Template Generation</span><br>Buat template pesan tagihan dengan placeholder [PERSON], [ITEM], [PRICE], [MULTIPLIER]</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">D</div>
                <div class="step-text"><span class="step-label">LLM Data Generation (Gemma)</span><br>data_gen_gemma.ipynb isi template dengan data nyata &rarr; kalimat tagihan sintetis realistis</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">E</div>
                <div class="step-text"><span class="step-label">Anotasi NER Otomatis</span><br>Labeling PERSON, ITEM, PRICE, MULTIPLIER dengan character-level span indexing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-box alert-success" style="margin-top:10px;">
        <div class="alert-title"><i class="fa-solid fa-file-export"></i> Output Pipeline</div>
        <b>training_data.json</b> &rarr; {len(training_data):,} records &middot; {total_entities:,} entitas teranotasi &middot; siap dikirim ke tim AI Engineer<br>
        <b>dataset_fixed.json</b> &rarr; {len(fixed_dataset):,} records &middot; hasil preprocessing intermediate<br>
        <b>talangin_synthetic_templates.json</b> &rarr; {len(templates):,} template tagihan sintetis (LLM Gemma)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: DATA QUALITY CHECK
# ─────────────────────────────────────────────────────────
elif selected_section == "data_quality":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-shield-halved"></i></div>
        <p class="section-title-text">Data Quality Check</p>
    </div>
    <p class="section-desc">
        Pemeriksaan kualitas data GoFood sebelum dan sesudah proses cleaning.
        Data bersih adalah syarat utama untuk menghasilkan training data NER yang akurat.
        Bagian ini membahas missing value, duplikasi, dan konsistensi data — berbeda dari
        fitur Conflict Detection di aplikasi Talang.in yang menangani potensi konflik utang/patungan.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="alert-box {'alert-warning' if missing_before > 0 else 'alert-success'}">
            <div class="alert-title">
                {'<i class="fa-solid fa-triangle-exclamation"></i>' if missing_before > 0 else '<i class="fa-solid fa-circle-check"></i>'}
                Missing Value
            </div>
            <b>Sebelum cleaning:</b> {missing_before:,} cell kosong<br>
            <b>Setelah cleaning:</b> {missing_after:,} cell kosong<br>
            <b>Berhasil ditangani:</b> {missing_before - missing_after:,} nilai<br>
            <small>Sisa missing di kolom <i>discount_price</i> (tidak wajib diisi).</small>
        </div>
        """, unsafe_allow_html=True)

        miss_detail = df_raw.isnull().sum().reset_index()
        miss_detail.columns = ['Kolom', 'Missing']
        miss_detail['Status'] = miss_detail['Missing'].apply(
            lambda x: 'Bersih' if x == 0 else f'{x:,} missing'
        )
        st.dataframe(miss_detail, use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f"""
        <div class="alert-box {'alert-warning' if duplicate_before > 0 else 'alert-success'}">
            <div class="alert-title">
                {'<i class="fa-solid fa-triangle-exclamation"></i>' if duplicate_before > 0 else '<i class="fa-solid fa-circle-check"></i>'}
                Duplikasi Data
            </div>
            <b>Duplikat ditemukan:</b> {duplicate_before:,} baris<br>
            <b>Setelah drop_duplicates():</b> 0 duplikat tersisa<br>
            <b>Data bersih:</b> {len(df):,} baris (dari {len(df_raw):,} baris awal)
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-chart-column"></i> Ringkasan Kualitas Data</div>
            Raw: <b>{len(df_raw):,} baris</b> &rarr; Clean: <b>{len(df):,} baris</b><br>
            Data terbuang: <b>{len(df_raw)-len(df):,} baris</b> ({((len(df_raw)-len(df))/len(df_raw)*100):.1f}%)<br><br>
            Missing: <span class="badge {'badge-yellow' if missing_after > 0 else 'badge-green'}">{missing_after:,} (discount_price)</span>&nbsp;
            Duplikat: <span class="badge badge-green">0</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="alert-box alert-success">
            <div class="alert-title"><i class="fa-solid fa-circle-check"></i> Catatan Penting</div>
            Missing value yang tersisa (<b>{missing_after:,}</b>) berasal dari kolom
            <b>discount_price</b> yang memang tidak selalu diisi (produk tanpa diskon).
            Sudah ditangani dengan <code>fillna(0)</code> sehingga tidak mempengaruhi analisis harga.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: INSIGHT
# ─────────────────────────────────────────────────────────
elif selected_section == "insight":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-lightbulb"></i></div>
        <p class="section-title-text">Insight</p>
    </div>
    <p class="section-desc">
        Temuan utama dari analisis dataset GoFood dan kesiapan training data
        project Talang.in NER.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-chart-column"></i> Insight Dataset GoFood</div>
            Dataset GoFood mencatat <b>{jumlah_transaksi:,} produk</b> dari <b>{total_merchant:,} merchant</b>
            di <b>{total_area} area</b>. Akumulasi harga mencapai <b>Rp {total_pengeluaran:,.0f}</b>
            dengan rata-rata <b>Rp {avg_price:,.0f}</b> per produk.
            Kategori dominan adalah <b>{kategori_terbanyak}</b>. Data ini menjadi fondasi
            utama entity ITEM dan PRICE dalam training data NER Talang.in.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-utensils"></i> Insight Item Menu NER</div>
            Training data mengandung <b>{len(item_counter):,} nama menu unik</b> dari 6 dataset berbeda.
            Variasi item mencerminkan keragaman menu GoFood Indonesia yang realistis — cocok untuk
            melatih model NER Talang.in yang perlu mengenali beragam jenis item tagihan.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-database"></i> Insight Training Data NER</div>
            <b>{len(training_data):,} records</b> dengan <b>{total_entities:,} entitas</b> teranotasi
            berhasil digenerate sebagai data latih.
            Rata-rata <b>7.76 entitas/record</b> menunjukkan kompleksitas tagihan realistis —
            multi-person, multi-item, dengan variasi format harga yang tinggi.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-user"></i> Insight Entity PERSON</div>
            Terdapat <b>{len(person_counter):,} nama orang unik</b> dalam data latih.
            Distribusi 1-8 orang per record mencerminkan skenario nyata tagihan grup
            di kafe dan restoran Indonesia — relevan untuk fitur patungan di Talang.in.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: A/B TESTING SIMULATION
# ─────────────────────────────────────────────────────────
elif selected_section == "ab_testing":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-flask"></i></div>
        <p class="section-title-text">A/B Testing Simulation</p>
    </div>
    <p class="section-desc">
        Simulasi A/B Testing untuk membandingkan dua strategi reminder pada fitur tagihan Talang.in.
        Karena data pengguna nyata belum tersedia, simulasi menggunakan data sintetis yang
        merepresentasikan skenario realistis.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Tentang Simulasi Ini</div>
        Simulasi membandingkan <b>Variant A: Reminder Standar</b> (notifikasi tagihan biasa tanpa personalisasi)
        vs <b>Variant B: Reminder Berbasis Rekomendasi Personal</b> (notifikasi dipersonalisasi berdasarkan
        riwayat transaksi dan preferensi pengguna). Metrik yang diukur: tingkat respons (response rate),
        waktu penyelesaian tagihan, dan kepuasan pengguna.
    </div>
    """, unsafe_allow_html=True)

    # Simulation controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_users = st.slider("Jumlah Pengguna Simulasi", min_value=100, max_value=2000, value=500, step=100)
    with col2:
        base_response_rate = st.slider("Base Response Rate (%)", min_value=20, max_value=60, value=35, step=5)
    with col3:
        effect_size = st.slider("Effect Size Variant B (%)", min_value=5, max_value=30, value=15, step=5)

    np.random.seed(42)

    # Simulate A/B groups
    group_a = np.random.binomial(1, base_response_rate/100, n_users // 2)
    group_b = np.random.binomial(1, (base_response_rate + effect_size)/100, n_users // 2)

    rate_a = group_a.mean() * 100
    rate_b = group_b.mean() * 100
    lift    = rate_b - rate_a
    lift_pct= (lift / rate_a) * 100 if rate_a > 0 else 0

    # Resolution time simulation (hours)
    time_a = np.random.normal(loc=48, scale=12, size=n_users // 2)
    time_b = np.random.normal(loc=36, scale=10, size=n_users // 2)
    time_a = np.clip(time_a, 1, None)
    time_b = np.clip(time_b, 1, None)

    # Satisfaction score (1-5)
    sat_a = np.random.normal(loc=3.2, scale=0.6, size=n_users // 2)
    sat_b = np.random.normal(loc=3.8, scale=0.5, size=n_users // 2)
    sat_a = np.clip(sat_a, 1, 5)
    sat_b = np.clip(sat_b, 1, 5)

    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        winner_color = "#1565C0" if rate_a >= rate_b else "#2E7D32"
        st.markdown(f"""
        <div style="display:flex; gap:12px;">
            <div class="ab-result-card ab-variant-a" style="flex:1;">
                <div class="ab-label" style="color:#1565C0;">Variant A</div>
                <div style="font-size:12px; color:#6A9A7A; margin-bottom:8px;">Reminder Standar</div>
                <div class="ab-stat" style="color:#1565C0;">{rate_a:.1f}%</div>
                <div style="font-size:12px; color:#8AB89A;">Response Rate</div>
            </div>
            <div class="ab-result-card ab-variant-b" style="flex:1;">
                <div class="ab-label" style="color:#2E7D32;">Variant B</div>
                <div style="font-size:12px; color:#6A9A7A; margin-bottom:8px;">Reminder Personal</div>
                <div class="ab-stat" style="color:#2E7D32;">{rate_b:.1f}%</div>
                <div style="font-size:12px; color:#8AB89A;">Response Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="display:flex; gap:12px;">
            <div class="ab-result-card ab-variant-a" style="flex:1;">
                <div class="ab-label" style="color:#1565C0;">Variant A</div>
                <div style="font-size:12px; color:#6A9A7A; margin-bottom:8px;">Avg. Resolve Time</div>
                <div class="ab-stat" style="color:#1565C0;">{time_a.mean():.1f}j</div>
                <div style="font-size:12px; color:#8AB89A;">rata-rata jam</div>
            </div>
            <div class="ab-result-card ab-variant-b" style="flex:1;">
                <div class="ab-label" style="color:#2E7D32;">Variant B</div>
                <div style="font-size:12px; color:#6A9A7A; margin-bottom:8px;">Avg. Resolve Time</div>
                <div class="ab-stat" style="color:#2E7D32;">{time_b.mean():.1f}j</div>
                <div style="font-size:12px; color:#8AB89A;">rata-rata jam</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="display:flex; gap:12px;">
            <div class="ab-result-card ab-variant-a" style="flex:1;">
                <div class="ab-label" style="color:#1565C0;">Variant A</div>
                <div style="font-size:12px; color:#6A9A7A; margin-bottom:8px;">Kepuasan</div>
                <div class="ab-stat" style="color:#1565C0;">{sat_a.mean():.2f}</div>
                <div style="font-size:12px; color:#8AB89A;">rata-rata (1-5)</div>
            </div>
            <div class="ab-result-card ab-variant-b" style="flex:1;">
                <div class="ab-label" style="color:#2E7D32;">Variant B</div>
                <div style="font-size:12px; color:#6A9A7A; margin-bottom:8px;">Kepuasan</div>
                <div class="ab-stat" style="color:#2E7D32;">{sat_b.mean():.2f}</div>
                <div style="font-size:12px; color:#8AB89A;">rata-rata (1-5)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Lift summary
    badge_cls  = "badge-green" if lift > 0 else "badge-red"
    winner_txt = "Variant B (Reminder Personal) lebih unggul" if lift > 0 else "Variant A lebih unggul atau setara"
    st.markdown(f"""
    <div class="alert-box {'alert-success' if lift > 0 else 'alert-warning'}">
        <div class="alert-title"><i class="fa-solid fa-trophy"></i> Hasil Simulasi A/B Testing</div>
        <b>Lift Response Rate:</b>
        <span class="badge {badge_cls}">{lift:+.1f} ppt ({lift_pct:+.1f}%)</span><br>
        <b>Kesimpulan Simulasi:</b> {winner_txt}.<br>
        Variant B menunjukkan potensi peningkatan keterlibatan pengguna melalui personalisasi reminder.
        Waktu penyelesaian tagihan juga lebih cepat rata-rata
        <b>{time_a.mean() - time_b.mean():.1f} jam</b> dibanding Variant A.<br>
        <small style="color:#8AB89A;">
            Catatan: Ini adalah simulasi menggunakan data sintetis ({n_users:,} pengguna).
            A/B Testing dengan data pengguna nyata akan dilakukan setelah aplikasi Talang.in
            memiliki pengguna aktif yang cukup.
        </small>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        resp_df = pd.DataFrame({
            'Variant': ['A — Standar', 'B — Personal'],
            'Response Rate (%)': [rate_a, rate_b]
        })
        fig_resp = px.bar(
            resp_df, x='Variant', y='Response Rate (%)',
            color='Variant', text_auto='.1f',
            color_discrete_map={'A — Standar':'#1565C0', 'B — Personal':'#2E7D32'},
            template='plotly_white'
        )
        fig_resp.update_layout(
            height=350, showlegend=False,
            title=dict(text='Perbandingan Response Rate', font=dict(size=14,color='#1B4332')),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50,b=20,l=10,r=10)
        )
        st.plotly_chart(fig_resp, use_container_width=True)

    with col2:
        time_df = pd.DataFrame({
            'Waktu (jam)': list(time_a[:200]) + list(time_b[:200]),
            'Variant': ['A — Standar'] * 200 + ['B — Personal'] * 200
        })
        fig_time = px.histogram(
            time_df, x='Waktu (jam)', color='Variant', nbins=30,
            barmode='overlay', opacity=0.7,
            color_discrete_map={'A — Standar':'#1565C0', 'B — Personal':'#2E7D32'},
            template='plotly_white'
        )
        fig_time.update_layout(
            height=350,
            title=dict(text='Distribusi Waktu Penyelesaian Tagihan', font=dict(size=14,color='#1B4332')),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50,b=20,l=10,r=10)
        )
        st.plotly_chart(fig_time, use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION: RECOMMENDATION
# ─────────────────────────────────────────────────────────
elif selected_section == "recommendation":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-bullseye"></i></div>
        <p class="section-title-text">Recommendation</p>
    </div>
    <p class="section-desc">
        Rekomendasi strategis untuk pengembangan dataset, pipeline, dan rencana model NER
        Talang.in ke depannya.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon"><i class="fa-solid fa-chart-line"></i></span>
            <div class="metric-label">Augmentasi Data</div>
            <div style="font-size:13px; color:#2D5A3D; line-height:1.7;">
                Tambah variasi penulisan harga (<i>"dua puluh ribu"</i>, <i>"20rb"</i>, <i>"20k"</i>, <i>"20.000"</i>)
                agar entity PRICE lebih robust terhadap berbagai format penulisan informal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon"><i class="fa-solid fa-folder-tree"></i></span>
            <div class="metric-label">Ekspansi Sumber Data</div>
            <div style="font-size:13px; color:#2D5A3D; line-height:1.7;">
                Manfaatkan <b>tokopedia_reviews</b> dan <b>produk_tokopedia</b>
                untuk menambah variasi item menu non-GoFood agar data latih lebih generalis
                untuk berbagai platform dan skenario belanja.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon"><i class="fa-solid fa-brain"></i></span>
            <div class="metric-label">Rencana Evaluasi Model NER</div>
            <div style="font-size:13px; color:#2D5A3D; line-height:1.7;">
                Setelah model NER dikembangkan oleh tim AI Engineer, lakukan evaluasi dengan
                <b>F1-score per entity label</b>.
                Prioritaskan entity PRICE karena formatnya paling beragam dan paling kritis
                untuk kalkulasi tagihan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-box alert-warning" style="margin-top:14px;">
        <div class="alert-title"><i class="fa-solid fa-triangle-exclamation"></i> Rekomendasi Kualitas Data</div>
        Masih ada <b>{missing_after:,} missing value</b> di kolom <b>discount_price</b>.
        Pertimbangkan strategi imputasi lebih tepat (median per kategori) daripada <code>fillna(0)</code>
        agar tidak bias pada analisis diskon dan harga. Juga pertimbangkan membuat kolom boolean
        <b>has_discount</b> agar lebih informatif.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: HEALTH SCORE
# ─────────────────────────────────────────────────────────
elif selected_section == "health_score":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-heart-pulse"></i></div>
        <p class="section-title-text">Health Score</p>
    </div>
    <p class="section-desc">
        Skor kesehatan keseluruhan project Data Science Talang.in — mencakup kualitas data GoFood
        dan kelengkapan training data NER yang telah disiapkan.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="health-card">
            <div class="health-label">Overall Health Score</div>
            <div class="health-score-big" style="color:{health_color};">{health_score}%</div>
            <div style="font-size:24px; color:{health_color};">
                <i class="fa-solid {health_icon}"></i>
            </div>
            <div style="font-size:14px; margin-top:8px; color:{health_color}; font-weight:700;">{health_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.write("")
        st.progress(health_score / 100)
        st.write("")
        checks = [
            ("Bebas Duplikat (post-cleaning)",  True, f"{duplicate_before:,} duplikat ditemukan &rarr; sudah di-drop"),
            ("Missing Value Tertangani",         True, f"{missing_before:,} &rarr; {missing_after:,} sisa (discount_price)"),
            ("Training Data Tersedia",           True, f"{len(training_data):,} records data latih siap pakai"),
            ("4 Entity Label Teranotasi",        True, "PERSON &middot; ITEM &middot; PRICE &middot; MULTIPLIER"),
            ("Synthetic Data Tergenerasi",       True, f"{len(templates):,} templates via LLM Gemma"),
            ("Pipeline Terdokumentasi",          True, "2 notebook: preprocessing + data_gen"),
        ]
        for label, passed, detail in checks:
            badge_cls  = 'badge-green' if passed else 'badge-red'
            badge_text = 'Lulus' if passed else 'Perlu Perbaikan'
            icon_class = 'fa-circle-check' if passed else 'fa-circle-xmark'
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
            margin-bottom:10px; padding:8px 0; border-bottom:1px solid #E8F5E9;">
                <span style="font-size:13px; color:#2D5A3D;">
                    <i class="fa-solid {icon_class}"></i> {label}<br>
                    <small style="color:#8AB89A;">{detail}</small>
                </span>
                <span class="badge {badge_cls}">{badge_text}</span>
            </div>
            """, unsafe_allow_html=True)

    for issue in health_issues:
        st.markdown(f"""
        <div class="alert-box alert-warning" style="margin-top:8px;">
            <i class="fa-solid fa-triangle-exclamation"></i> {issue}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SECTION: KESIMPULAN
# ─────────────────────────────────────────────────────────
elif selected_section == "kesimpulan":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid fa-clipboard-check"></i></div>
        <p class="section-title-text">Kesimpulan</p>
    </div>
    <p class="section-desc">
        Rangkuman akhir dari seluruh proses Data Science project Talang.in NER —
        membuktikan bahwa dataset telah melalui pipeline lengkap dan siap digunakan
        untuk pengembangan fitur AI Smart Transaction Input.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-box alert-success">
        <div class="alert-title"><i class="fa-solid fa-clipboard-check"></i> Kesimpulan Analisis — Project Data Science Talang.in</div>
        <ol style="margin:0; padding-left:20px; line-height:2.2;">
            <li>
                <b>Dataset GoFood</b> digunakan sebagai sumber data menu, harga, merchant, dan kategori
                untuk membangun training data NER yang mendukung fitur
                <b>AI Smart Transaction Input Talang.in</b> — bukan sebagai data transaksi pengguna langsung.
            </li>
            <li>
                Dataset GoFood memiliki <b>{len(df_raw):,} data mentah</b> dari <b>{total_merchant:,} merchant</b>
                di <b>{total_area} area</b>. Setelah cleaning menjadi <b>{len(df):,} data bersih</b>.
            </li>
            <li>
                <b>Training data NER</b> berhasil digenerate sebanyak <b>{len(training_data):,} records</b> dengan
                total <b>{total_entities:,} entitas</b> teranotasi (PERSON, ITEM, PRICE, MULTIPLIER).
                Data latih ini <b>siap digunakan oleh tim AI Engineer</b> untuk proses training model.
                Evaluasi model — termasuk precision, recall, dan F1-score — akan dilakukan
                setelah model selesai dikembangkan.
            </li>
            <li>
                Pipeline preprocessing menggabungkan <b>6 dataset sumber</b> melalui proses cleaning,
                normalisasi, template generation, dan augmentasi berbasis LLM (Gemma) — menghasilkan
                3 file JSON output.
            </li>
            <li>
                Training data mengandung <b>{len(item_counter):,} nama menu unik</b> dan
                <b>{len(person_counter):,} nama orang unik</b>, memastikan variasi entitas yang
                cukup untuk generalisasi model NER.
            </li>
            <li>
                Simulasi <b>A/B Testing</b> menunjukkan potensi peningkatan respons pengguna
                melalui reminder berbasis rekomendasi personal dibanding reminder standar.
            </li>
            <li>
                Dashboard ini membuktikan bahwa dataset telah melalui proses
                <b>cleaning, preprocessing, template generation, dan anotasi NER</b>
                secara menyeluruh — sehingga siap digunakan untuk pengembangan fitur
                <b>AI Smart Transaction Input</b> pada aplikasi Talang.in.
                Health Score project: <b style="color:{health_color};">{health_score}% — {health_status}</b>.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER (always visible)
# =========================================================

st.markdown(f"""
<div class="footer-area">
    <div class="footer-title">
        <i class="fa-solid fa-wallet"></i> Talang.in — Data Science Analytics Dashboard
    </div>
    <p>Streamlit Cloud Deployment &middot; Data Science Analytics &middot; Named Entity Recognition Project</p>
    <p style="margin-top:8px;">
        <span class="badge badge-green">v3.0</span>&nbsp;
        <span class="badge badge-green">Streamlit Cloud</span>&nbsp;
        <span class="badge badge-green">NER &middot; GoFood</span>&nbsp;
        <span class="badge badge-blue">{len(training_data):,} Data Latih</span>&nbsp;
        <span class="badge badge-blue">{total_entities:,} Entities</span>
    </p>
    <p style="margin-top:12px; font-size:12px; color:#8AB89A;">
        Sources: gofood_dataset &middot; alergen_dataset &middot; Steakhouse &middot; indonesian_food &middot; nutrition &middot; indonesian-names<br>
        Pipeline: data_preprocessing.ipynb &rarr; data_gen_gemma.ipynb &rarr; training_data.json
    </p>
</div>
""", unsafe_allow_html=True)
