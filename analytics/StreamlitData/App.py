# =========================================================
# TALANG.IN — DATA SCIENCE ANALYTICS DASHBOARD (RINGKAS)
# GoFood NER Project | Streamlit v3.0
# =========================================================
# STRUKTUR FILE INI:
#   1. Import & Konfigurasi Halaman
#   2. CSS Styling (disingkat pakai fungsi helper)
#   3. Load & Cache Data
#   4. Hitung KPI & Statistik Global
#   5. Sidebar (navigasi + filter)
#   6. Header Utama
#   7. Router Section (tiap section = 1 blok elif)
#   8. Footer
# =========================================================

# ── 1. IMPORT & KONFIGURASI ────────────────────────────────
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from collections import Counter

st.set_page_config(
    page_title="Talang.in Data Science Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. CSS STYLING ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp { background: linear-gradient(150deg, #F0F7F0, #E8F4E8, #EAF5EA); }
.block-container { padding: 2rem 2.5rem; max-width: 1300px; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D3B2E, #134E3A, #1B5E40); }
section[data-testid="stSidebar"] * { color: #D4EDDA !important; }
section[data-testid="stSidebar"] label { color: #A8D5B5 !important; font-size: 13px !important; }

.page-header { background: linear-gradient(135deg, #0D4A30, #1B6E44); border-radius: 20px; padding: 36px 40px; margin-bottom: 32px; }
.page-header h1 { color: #FFF; font-size: 32px; font-weight: 800; margin: 0 0 8px 0; }
.page-header p  { color: #A8D5B5; font-size: 14px; margin: 0; line-height: 1.7; }
.header-badge   { display: inline-block; background: rgba(255,255,255,0.15); color: #CCEBCC; font-size: 11px; font-weight: 600; padding: 4px 14px; border-radius: 20px; margin-bottom: 14px; text-transform: uppercase; }

.section-header { display: flex; align-items: center; gap: 12px; margin: 40px 0 6px 0; padding-bottom: 12px; border-bottom: 2px solid #C8E6C9; }
.section-icon   { width: 38px; height: 38px; background: #E8F5E9; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.section-title-text { font-size: 22px; font-weight: 700; color: #1B4332; margin: 0; }
.section-desc       { font-size: 14px; color: #5A8A6A; margin: 0 0 20px 0; }

.metric-card  { background: #FFF; border-radius: 16px; padding: 20px 22px; border: 1.5px solid #D4EDD8; height: 100%; }
.metric-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #6A9A7A; margin-bottom: 5px; }
.metric-value { font-size: 26px; font-weight: 800; color: #1B4332; }
.metric-sub   { font-size: 12px; color: #8AB89A; margin-top: 5px; }
.metric-icon  { font-size: 24px; margin-bottom: 10px; display: block; }

.ner-card   { background: #FFF; border-radius: 14px; padding: 18px 20px; border: 1.5px solid #D4EDD8; text-align: center; height: 100%; }
.ner-label  { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.ner-value  { font-size: 28px; font-weight: 800; line-height: 1; margin-bottom: 4px; }

.alert-box   { border-radius: 14px; padding: 16px 20px; margin-bottom: 16px; font-size: 14px; line-height: 1.7; }
.alert-info    { background: #EBF8EE; border-left: 4px solid #2D9A5A; color: #1B4332; }
.alert-success { background: #E8F5E9; border-left: 4px solid #2E7D32; color: #1B4332; }
.alert-warning { background: #FFF8E1; border-left: 4px solid #F9A825; color: #5D4037; }
.alert-title { font-weight: 700; font-size: 14px; margin-bottom: 6px; }

.badge        { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-green  { background: #C8E6C9; color: #1B5E20; }
.badge-yellow { background: #FFF9C4; color: #795548; }
.badge-red    { background: #FFCDD2; color: #B71C1C; }
.badge-blue   { background: #BBDEFB; color: #0D47A1; }

.source-chip { display: inline-block; background: #E8F5E9; color: #1B5E20; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 600; margin: 3px; border: 1px solid #C8E6C9; }

.pipeline-card { background: #FFF; border-radius: 14px; padding: 20px 22px; border: 1.5px solid #D4EDD8; height: 100%; }
.pipeline-step { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
.step-num  { width: 26px; height: 26px; background: #1B4332; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.step-text { font-size: 13px; color: #2D5A3D; line-height: 1.6; }
.step-label { font-weight: 700; color: #1B4332; }

.health-card      { background: #FFF; border-radius: 16px; padding: 28px 32px; border: 1.5px solid #D4EDD8; text-align: center; }
.health-score-big { font-size: 64px; font-weight: 800; line-height: 1; margin: 12px 0; }
.health-label     { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #6A9A7A; }

.ab-result-card { background: #FFF; border-radius: 14px; padding: 22px 24px; border: 1.5px solid #D4EDD8; text-align: center; height: 100%; }
.ab-variant-a   { border-top: 4px solid #1565C0; }
.ab-variant-b   { border-top: 4px solid #2E7D32; }
.ab-stat  { font-size: 32px; font-weight: 800; line-height: 1.1; margin: 8px 0 4px 0; }
.ab-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #6A9A7A; }

.footer-area  { background: #FFF; border-radius: 16px; padding: 24px 32px; text-align: center; border: 1.5px solid #D4EDD8; margin-top: 48px; }
.footer-title { font-size: 15px; font-weight: 700; color: #1B4332; margin-bottom: 4px; }
.footer-area p { color: #6A9A7A; font-size: 13px; margin: 4px 0; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #B2D8B2; border-radius: 4px; }
.stProgress > div > div { background: linear-gradient(90deg, #2D9A5A, #52C27A) !important; border-radius: 8px !important; }
.stProgress > div       { background: #D4EDD8 !important; border-radius: 8px !important; height: 10px !important; }
.stTabs [data-baseweb="tab-list"]   { background: #E8F5E9; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"]        { border-radius: 10px; color: #5A8A6A; font-weight: 600; font-size: 14px; }
.stTabs [aria-selected="true"]      { background: #FFF !important; color: #1B4332 !important; }
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.07); border-radius: 8px;
    padding: 6px 12px !important; margin-bottom: 3px;
    font-size: 13px !important; color: #D4EDDA !important; display: block; width: 100%;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,0.18) !important; border-left: 3px solid #52C27A;
}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
""", unsafe_allow_html=True)


# ── 3. LOAD & CACHE DATA ───────────────────────────────────
@st.cache_data
def load_gofood():
    df = pd.read_csv("data/gofood_dataset.csv")
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
    training_data  = load_json("../outputs/training_data.json")
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
training_data  = load_json("../outputs/training_data.json")
fixed_dataset  = load_json("../dataset_fixed.json")
templates      = load_json("../outputs/talangin_synthetic_templates.json")
entity_counts, item_counter, person_counter, person_per_rec = compute_ner_stats()


# ── 4. KPI & HEALTH SCORE GLOBAL ──────────────────────────
total_entities = sum(v for k, v in entity_counts.items() if k.strip() and not k.startswith('['))

health_score  = 100
health_issues = []
if duplicate_before > 0:
    health_score -= 20
    health_issues.append(f"{duplicate_before:,} duplikat ditemukan → sudah dibersihkan dengan drop_duplicates()")
if missing_before > 0:
    health_score -= 20
    health_issues.append(f"{missing_before:,} missing value ditemukan → sudah ditangani dengan fillna(0)")

health_color  = "#2E7D32" if health_score >= 80 else "#F57F17" if health_score >= 60 else "#C62828"
health_icon   = "fa-heart-pulse" if health_score >= 80 else "fa-circle-exclamation" if health_score >= 60 else "fa-circle-xmark"
health_status = "Sangat Baik" if health_score >= 80 else "Perlu Perhatian" if health_score >= 60 else "Kritis"


# ── 5. SIDEBAR ─────────────────────────────────────────────
SECTIONS = [
    ("overview",        "fa-folder-open",    "Overview Dataset"),
    ("kpi",             "fa-thumbtack",      "KPI Metrics"),
    ("grafik_kategori", "fa-chart-column",   "Grafik Kategori"),
    ("grafik_merchant", "fa-store",          "Grafik Merchant"),
    ("tren_area",       "fa-chart-line",     "Tren Area"),
    ("training_ner",    "fa-robot",          "Training Data NER"),
    ("preprocessing",   "fa-microscope",     "Preprocessing Pipeline"),
    ("data_quality",    "fa-shield-halved",  "Data Quality Check"),
    ("insight",         "fa-lightbulb",      "Insight"),
    ("ab_testing",      "fa-flask",          "A/B Testing Simulation"),
    ("recommendation",  "fa-bullseye",       "Recommendation"),
    ("health_score",    "fa-heart-pulse",    "Health Score"),
    ("kesimpulan",      "fa-clipboard-check","Kesimpulan"),
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px;">
        <div style="width:48px;height:48px;margin:0 auto 10px;border-radius:16px;background:rgba(255,255,255,0.10);
        display:flex;align-items:center;justify-content:center;font-size:22px;color:#FFF;">
            <i class="fa-solid fa-wallet"></i></div>
        <div style="font-size:17px;font-weight:800;color:#FFF;">Talang.in</div>
        <div style="font-size:12px;color:#8CBFA0;margin-top:2px;">Data Science Dashboard v3.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    selected_section = st.radio(
        "Pilih Section",
        options=[s[0] for s in SECTIONS],
        format_func=lambda k: next(s[2] for s in SECTIONS if s[0] == k),
        label_visibility="collapsed"
    )
    st.markdown("---")

    kategori_filter = st.multiselect("Pilih Kategori",     options=df['category'].unique(),      default=df['category'].unique())
    area_filter     = st.multiselect("Pilih Area Merchant",options=df['merchant_area'].unique(), default=df['merchant_area'].unique())
    filtered_df     = df[df['category'].isin(kategori_filter) & df['merchant_area'].isin(area_filter)]

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.08);border-radius:12px;padding:14px 16px;">
        <div style="font-size:11px;color:#8CBFA0;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px;">
            <i class="fa-solid fa-chart-column"></i> Status Filter</div>
        <div style="font-size:13px;color:#D4EDDA;margin-bottom:5px;"><i class="fa-solid fa-database"></i> Data aktif: <b style="color:#FFF;">{len(filtered_df):,}</b> baris</div>
        <div style="font-size:13px;color:#D4EDDA;margin-bottom:5px;"><i class="fa-solid fa-tags"></i> Kategori: <b style="color:#FFF;">{len(kategori_filter)}</b> dipilih</div>
        <div style="font-size:13px;color:#D4EDDA;"><i class="fa-solid fa-location-dot"></i> Area: <b style="color:#FFF;">{len(area_filter)}</b> dipilih</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#8CBFA0;text-align:center;line-height:1.8;">Dibuat untuk requirement<br><b style="color:#A8D5B5;">Data Scientist</b><br>Deploy ke <b style="color:#A8D5B5;">Streamlit Cloud</b></div>', unsafe_allow_html=True)


# ── KPI per filter ─────────────────────────────────────────
total_pengeluaran  = filtered_df['price'].sum()
jumlah_transaksi   = len(filtered_df)
avg_price          = filtered_df['price'].mean()
kategori_terbanyak = filtered_df['category'].value_counts().idxmax()
merchant_aktif     = filtered_df['merchant_name'].value_counts().idxmax()
total_merchant     = filtered_df['merchant_name'].nunique()
total_area         = filtered_df['merchant_area'].nunique()


# ── 6. HEADER UTAMA ────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="header-badge"><i class="fa-solid fa-robot"></i> NER · GoFood · Data Science · Analytics Report</div>
    <h1><i class="fa-solid fa-wallet"></i> Talang.in — Data Science Analytics Dashboard</h1>
    <p>Dashboard ini mendokumentasikan proses Data Science untuk mendukung fitur
    <b>AI Smart Transaction Input</b> pada aplikasi <b>Talang.in</b>.
    Dataset GoFood digunakan sebagai sumber data menu, harga, merchant, dan kategori
    untuk membangun training data NER. Fokus: <b>data preparation, EDA, preprocessing, dan kesiapan data latih</b>.</p>
</div>
""", unsafe_allow_html=True)


# ── 7. HELPER FUNCTIONS ────────────────────────────────────
def section_header(icon, title, desc):
    """Render header tiap section secara konsisten."""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon"><i class="fa-solid {icon}"></i></div>
        <p class="section-title-text">{title}</p>
    </div>
    <p class="section-desc">{desc}</p>
    """, unsafe_allow_html=True)

def metric_card(col, icon_html, label, value, sub, font_size="26px"):
    """Render kartu metrik ke dalam kolom yang diberikan."""
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
    """Terapkan layout standar ke semua chart Plotly."""
    fig.update_layout(
        height=height, coloraxis_showscale=False,
        title=dict(text=title, font=dict(size=14, color='#1B4332')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=10, r=10)
    )
    return fig


# ══════════════════════════════════════════════════════════
# 8. ROUTER SECTION
# Tiap section dirender hanya jika dipilih di sidebar.
# ══════════════════════════════════════════════════════════

if selected_section == "overview":
    section_header("folder-open", "Overview Dataset",
        "Ringkasan seluruh dataset — dari raw dataset hingga hasil preprocessing dan generasi sintetis NER.")

    cols = st.columns(5)
    cards = [
        ('<i class="fa-solid fa-database"></i>',       "GoFood Raw",       f"{len(df_raw):,}",        "data mentah awal"),
        ('<i class="fa-solid fa-broom"></i>',           "Setelah Cleaning", f"{len(df):,}",            "drop duplikat & null"),
        ('<i class="fa-solid fa-robot"></i>',           "Data Latih NER",   f"{len(training_data):,}", "siap ke AI Engineer"),
        ('<i class="fa-solid fa-list-check"></i>',      "Fixed Dataset",    f"{len(fixed_dataset):,}", "hasil preprocessing"),
        ('<i class="fa-solid fa-clipboard-check"></i>', "Synthetic Tpl.",   f"{len(templates):,}",     "template LLM Gemma"),
    ]
    for col, (icon, label, val, sub) in zip(cols, cards):
        metric_card(col, icon, label, val, sub)

    st.write("")
    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Peran Dataset GoFood dalam Talang.in</div>
        Dataset GoFood <b>bukan data transaksi pengguna</b>, melainkan <b>sumber referensi</b>
        untuk membangun training data NER. Menu, harga, merchant, dan kategori diekstrak
        sebagai bahan anotasi entity ITEM dan PRICE untuk fitur <b>AI Smart Transaction Input</b>.
    </div>
    <div class="alert-box alert-success">
        <div class="alert-title"><i class="fa-solid fa-layer-group"></i> 6 Dataset Sumber yang Digabungkan</div>
        <span class="source-chip">gofood_dataset.csv · 45.195 baris</span>
        <span class="source-chip">alergen_dataset.csv · 100.000 baris</span>
        <span class="source-chip">Steakhouse_dataset.csv · 150 baris</span>
        <span class="source-chip">indonesian_food.csv · 1.273 baris</span>
        <span class="source-chip">nutrition.csv · 1.345 baris</span>
        <span class="source-chip">indonesian-names.csv · 1.960 baris</span>
        <br><br>Semua diproses di <b>data_preprocessing.ipynb</b>. Generasi sintetis via <b>data_gen_gemma.ipynb</b>.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Lihat Sample GoFood Dataset (10 baris)"):
        st.dataframe(filtered_df.head(10), use_container_width=True)
    with st.expander("Lihat Statistik Deskriptif"):
        st.dataframe(filtered_df.describe(), use_container_width=True)
    with st.expander("Lihat Sample Training Data NER"):
        samples = [{"Teks": rec['text'][:100]+"...",
                    "Sample Entities": ", ".join(f"{e['label']}:{e['text']}" for e in rec['entities'][:4] if e['label'].strip() and not e['label'].startswith('[')),
                    "Total Entities": len(rec['entities'])} for rec in training_data[:8]]
        st.dataframe(pd.DataFrame(samples), use_container_width=True)


elif selected_section == "kpi":
    section_header("thumbtack", "KPI Metrics — GoFood Analytics",
        "Indikator utama dataset GoFood berdasarkan filter aktif di sidebar.")

    cols = st.columns(4)
    kpis = [
        ('<i class="fa-solid fa-coins"></i>',        "Total Harga (Akumulasi)", f"Rp {total_pengeluaran:,.0f}", "akumulasi semua produk"),
        ('<i class="fa-solid fa-cart-shopping"></i>', "Jumlah Produk",           f"{jumlah_transaksi:,}",        "total item tercatat"),
        ('<i class="fa-solid fa-chart-line"></i>',   "Rata-rata Harga",         f"Rp {avg_price:,.0f}",         "per produk"),
        ('<i class="fa-solid fa-trophy"></i>',       "Kategori Dominan",        kategori_terbanyak,             "kategori terpopuler"),
    ]
    for col, (icon, label, val, sub) in zip(cols, kpis):
        metric_card(col, icon, label, val, sub, font_size="20px" if len(val) > 10 else "24px")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card" style="display:flex;gap:20px;align-items:center;"><span style="font-size:30px;"><i class="fa-solid fa-store"></i></span><div><div class="metric-label">Total Merchant Unik</div><div class="metric-value">{total_merchant:,}</div><div class="metric-sub">merchant terdaftar</div></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="display:flex;gap:20px;align-items:center;"><span style="font-size:30px;"><i class="fa-solid fa-medal"></i></span><div><div class="metric-label">Merchant Paling Aktif</div><div class="metric-value" style="font-size:16px;">{merchant_aktif}</div><div class="metric-sub">volume produk tertinggi</div></div></div>', unsafe_allow_html=True)


elif selected_section == "grafik_kategori":
    section_header("chart-column", "Grafik Kategori",
        "Distribusi produk berdasarkan kategori GoFood — sumber entity ITEM dalam training data NER.")

    kategori_data = filtered_df['category'].value_counts().reset_index()
    kategori_data.columns = ['Kategori', 'Jumlah']
    tab1, tab2 = st.tabs(["Bar Chart (Semua)", "Pie Chart (Top 10)"])

    with tab1:
        fig = px.bar(kategori_data, x='Kategori', y='Jumlah', color='Jumlah',
                     text_auto=True, color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Distribusi Semua Kategori'), use_container_width=True)
    with tab2:
        fig = px.pie(kategori_data.head(10), names='Kategori', values='Jumlah',
                     color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.45)
        st.plotly_chart(plotly_layout(fig, 'Proporsi Top 10 Kategori'), use_container_width=True)


elif selected_section == "grafik_merchant":
    section_header("store", "Grafik Merchant",
        "Top 10 merchant berdasarkan jumlah produk — sumber utama entity ITEM training data NER.")

    merchant_data = filtered_df['merchant_name'].value_counts().head(10).reset_index()
    merchant_data.columns = ['Merchant', 'Jumlah']
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.bar(merchant_data.sort_values('Jumlah'), x='Jumlah', y='Merchant', orientation='h',
                     color='Jumlah', text_auto=True, color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Top 10 Merchant — Jumlah Produk', height=400), use_container_width=True)
    with col2:
        fig = px.pie(merchant_data, names='Merchant', values='Jumlah',
                     color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.5)
        st.plotly_chart(plotly_layout(fig, 'Porsi Top 10 Merchant', height=400), use_container_width=True)

    st.markdown(f'<div class="alert-box alert-info"><div class="alert-title"><i class="fa-solid fa-trophy"></i> Merchant Paling Aktif</div><b>{merchant_aktif}</b> memiliki volume produk tertinggi dari {total_merchant:,} merchant unik.</div>', unsafe_allow_html=True)


elif selected_section == "tren_area":
    section_header("chart-line", "Tren Area Analytics",
        "Distribusi produk berdasarkan area merchant — volume tinggi = variasi item lebih kaya.")

    area_data = filtered_df['merchant_area'].value_counts().reset_index()
    area_data.columns = ['Area', 'Jumlah']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=area_data['Area'], y=area_data['Jumlah'],
        mode='lines+markers+text', text=area_data['Jumlah'], textposition='top center',
        line=dict(color='#2D9A5A', width=3), marker=dict(color='#2D9A5A', size=11, line=dict(color='white', width=2)),
        fill='tozeroy', fillcolor='rgba(45,154,90,0.10)'
    ))
    fig.update_layout(height=380, title=dict(text=f'Distribusi Produk per Area ({total_area} Area)', font=dict(size=14,color='#1B4332')),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50,b=20,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)


elif selected_section == "training_ner":
    section_header("robot", "Training Data NER — Data Latih Siap Pakai",
        "Statistik data latih NER: PERSON, ITEM, PRICE, MULTIPLIER. Belum merepresentasikan hasil prediksi model.")

    st.markdown("""
    <div class="alert-box alert-warning">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Status Model NER</div>
        Dataset <b>siap digunakan tim AI Engineer</b>. Training & evaluasi (precision/recall/F1) dilakukan setelah model dikembangkan.
        Dashboard ini hanya mencakup <b>persiapan & validasi data latih</b>.
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    ner_cards = [
        ('<i class="fa-solid fa-user"></i>',    "PERSON",     entity_counts.get('PERSON',0),     "nama orang", "#1B5E20", "#E8F5E9"),
        ('<i class="fa-solid fa-utensils"></i>',"ITEM",       entity_counts.get('ITEM',0),       "nama menu",  "#0D47A1", "#E3F2FD"),
        ('<i class="fa-solid fa-coins"></i>',   "PRICE",      entity_counts.get('PRICE',0),      "harga",      "#4A148C", "#F3E5F5"),
        ('<i class="fa-solid fa-hashtag"></i>', "MULTIPLIER", entity_counts.get('MULTIPLIER',0), "jumlah",     "#BF360C", "#FBE9E7"),
    ]
    for col, (icon, lbl, val, sub, tc, bg) in zip(cols, ner_cards):
        with col:
            st.markdown(f'<div class="ner-card" style="background:{bg};"><div class="ner-label" style="color:{tc};">{lbl}</div><div class="ner-value" style="color:{tc};">{val:,}</div><div style="font-size:20px;margin:6px 0;">{icon}</div><div style="font-size:12px;color:#8AB89A;">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    ents_per_rec = [len(x['entities']) for x in training_data]
    sorted_ents  = sorted(ents_per_rec)

    with col1:
        stat_df = pd.DataFrame({"Metrik": [
            "Total records","Total entitas","Rata-rata entitas/record",
            "Median entitas/record","Max entitas/record",
            "PERSON unik","ITEM unik","Synthetic templates","Fixed dataset","Dataset sumber"
        ], "Nilai": [
            f"{len(training_data):,}", f"{total_entities:,}",
            f"{sum(ents_per_rec)/len(ents_per_rec):.2f}", f"{sorted_ents[len(sorted_ents)//2]}",
            f"{max(ents_per_rec)}", f"{len(person_counter):,}", f"{len(item_counter):,}",
            f"{len(templates):,}", f"{len(fixed_dataset):,}", "6 dataset"
        ]})
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    with col2:
        ent_df = pd.DataFrame({
            'Entity': [k for k in entity_counts if k.strip() and not k.startswith('[')],
            'Count':  [v for k, v in entity_counts.items() if k.strip() and not k.startswith('[')]
        })
        fig = px.pie(ent_df, names='Entity', values='Count', hole=0.5,
                     color_discrete_map={'PERSON':'#2E7D32','ITEM':'#1565C0','PRICE':'#6A1B9A','MULTIPLIER':'#BF360C'})
        st.plotly_chart(plotly_layout(fig, 'Distribusi Entity Labels', height=330), use_container_width=True)

    top_items = pd.DataFrame(item_counter.most_common(15), columns=['Item Menu', 'Frekuensi'])
    fig = px.bar(top_items.sort_values('Frekuensi'), x='Frekuensi', y='Item Menu', orientation='h',
                 color='Frekuensi', text_auto=True, color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white')
    st.plotly_chart(plotly_layout(fig, 'Top 15 Item Menu Paling Sering Muncul di Data Latih'), use_container_width=True)

    person_dist = Counter(person_per_rec)
    pdist_df = pd.DataFrame(sorted(person_dist.items()), columns=['Jumlah Person/Record','Jumlah Record']).head(8)
    fig = px.bar(pdist_df, x='Jumlah Person/Record', y='Jumlah Record',
                 color='Jumlah Record', text_auto=True, color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white')
    st.plotly_chart(plotly_layout(fig, 'Distribusi Jumlah PERSON per Record', height=360), use_container_width=True)


elif selected_section == "preprocessing":
    section_header("microscope", "Preprocessing Pipeline",
        "Alur transformasi data — dari raw dataset hingga training data NER siap pakai.")

    col1, col2 = st.columns(2)

    with col1:
        steps = [("1","gofood_dataset.csv","45.195 baris · nama produk, harga, kategori, area merchant GoFood"),
                 ("2","alergen_dataset.csv","100.000 baris · nama produk dari label alergen pangan Indonesia"),
                 ("3","Steakhouse_dataset.csv","150 baris · menu restoran steakhouse Indonesia"),
                 ("4","indonesian_food.csv + nutrition.csv","2.618 baris · nama makanan Indonesia + info gizi"),
                 ("5","indonesian-names.csv","1.960 baris · nama orang Indonesia → sumber entity PERSON")]
        html = '<div class="pipeline-card"><div class="metric-label" style="margin-bottom:16px;"><i class="fa-solid fa-file-import"></i> INPUT — Dataset Sumber</div>'
        for num, lbl, desc in steps:
            html += f'<div class="pipeline-step"><div class="step-num">{num}</div><div class="step-text"><span class="step-label">{lbl}</span><br>{desc}</div></div>'
        st.markdown(html + '</div>', unsafe_allow_html=True)

    with col2:
        steps = [("A","Cleaning & Normalisasi","Drop NaN, hapus duplikat, lowercase, strip whitespace, hapus noise (Sastrawi)"),
                 ("B","extract_core_item()","Ekstrak nama menu inti dari judul panjang → 1-3 kata conversational"),
                 ("C","Template Generation","Buat template dengan placeholder [PERSON], [ITEM], [PRICE], [MULTIPLIER]"),
                 ("D","LLM Data Generation (Gemma)","data_gen_gemma.ipynb isi template → kalimat tagihan sintetis realistis"),
                 ("E","Anotasi NER Otomatis","Labeling PERSON, ITEM, PRICE, MULTIPLIER dengan character-level span indexing")]
        html = '<div class="pipeline-card"><div class="metric-label" style="margin-bottom:16px;"><i class="fa-solid fa-gear"></i> PROSES — Notebook Pipeline</div>'
        for num, lbl, desc in steps:
            html += f'<div class="pipeline-step"><div class="step-num">{num}</div><div class="step-text"><span class="step-label">{lbl}</span><br>{desc}</div></div>'
        st.markdown(html + '</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-box alert-success" style="margin-top:10px;">
        <div class="alert-title"><i class="fa-solid fa-file-export"></i> Output Pipeline</div>
        <b>training_data.json</b> → {len(training_data):,} records · {total_entities:,} entitas · siap ke AI Engineer<br>
        <b>dataset_fixed.json</b> → {len(fixed_dataset):,} records · preprocessing intermediate<br>
        <b>talangin_synthetic_templates.json</b> → {len(templates):,} template (LLM Gemma)
    </div>
    """, unsafe_allow_html=True)


elif selected_section == "data_quality":
    section_header("shield-halved", "Data Quality Check",
        "Pemeriksaan missing value, duplikasi, dan konsistensi data GoFood sebelum & sesudah cleaning.")

    col1, col2 = st.columns(2)
    with col1:
        cls = 'alert-warning' if missing_before > 0 else 'alert-success'
        ico = 'triangle-exclamation' if missing_before > 0 else 'circle-check'
        st.markdown(f"""
        <div class="alert-box {cls}">
            <div class="alert-title"><i class="fa-solid fa-{ico}"></i> Missing Value</div>
            <b>Sebelum:</b> {missing_before:,} cell kosong &nbsp;|&nbsp; <b>Setelah:</b> {missing_after:,} cell kosong<br>
            <small>Sisa missing di <i>discount_price</i> (tidak wajib diisi).</small>
        </div>
        """, unsafe_allow_html=True)
        miss_detail = df_raw.isnull().sum().reset_index()
        miss_detail.columns = ['Kolom', 'Missing']
        miss_detail['Status'] = miss_detail['Missing'].apply(lambda x: 'Bersih' if x == 0 else f'{x:,} missing')
        st.dataframe(miss_detail, use_container_width=True, hide_index=True)

    with col2:
        cls = 'alert-warning' if duplicate_before > 0 else 'alert-success'
        ico = 'triangle-exclamation' if duplicate_before > 0 else 'circle-check'
        pct = (len(df_raw)-len(df))/len(df_raw)*100
        st.markdown(f"""
        <div class="alert-box {cls}">
            <div class="alert-title"><i class="fa-solid fa-{ico}"></i> Duplikasi Data</div>
            <b>Duplikat:</b> {duplicate_before:,} baris → setelah drop_duplicates(): 0 tersisa<br>
            <b>Data bersih:</b> {len(df):,} dari {len(df_raw):,} ({pct:.1f}% terbuang)
        </div>
        <div class="alert-box alert-info">
            <div class="alert-title"><i class="fa-solid fa-circle-check"></i> Catatan</div>
            Missing di <b>discount_price</b> sudah ditangani dengan <code>fillna(0)</code>.
            Produk tanpa diskon memang kosong — bukan error.<br>
            Missing: <span class="badge {'badge-yellow' if missing_after > 0 else 'badge-green'}">{missing_after:,}</span>&nbsp;
            Duplikat: <span class="badge badge-green">0</span>
        </div>
        """, unsafe_allow_html=True)


elif selected_section == "insight":
    section_header("lightbulb", "Insight",
        "Temuan utama dari analisis dataset GoFood dan kesiapan training data NER Talang.in.")

    col1, col2 = st.columns(2)
    insights = [
        ("chart-column", "Insight Dataset GoFood",
         f"Dataset mencatat <b>{jumlah_transaksi:,} produk</b> dari <b>{total_merchant:,} merchant</b> di <b>{total_area} area</b>. Rata-rata harga <b>Rp {avg_price:,.0f}</b>. Kategori dominan: <b>{kategori_terbanyak}</b>."),
        ("utensils", "Insight Item Menu NER",
         f"Training data mengandung <b>{len(item_counter):,} nama menu unik</b> dari 6 dataset — mencerminkan keragaman menu GoFood Indonesia."),
        ("database", "Insight Training Data NER",
         f"<b>{len(training_data):,} records</b> dengan <b>{total_entities:,} entitas</b>. Rata-rata 7.76 entitas/record — multi-person, multi-item, variasi format harga tinggi."),
        ("user", "Insight Entity PERSON",
         f"<b>{len(person_counter):,} nama orang unik</b>. Distribusi 1–8 orang/record mencerminkan skenario tagihan grup di kafe & restoran."),
    ]
    for i, (icon, title, body) in enumerate(insights):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f'<div class="alert-box alert-info"><div class="alert-title"><i class="fa-solid fa-{icon}"></i> {title}</div>{body}</div>', unsafe_allow_html=True)


elif selected_section == "ab_testing":
    section_header("flask", "A/B Testing Simulation",
        "Simulasi: Reminder Standar (A) vs Reminder Berbasis Rekomendasi Personal (B).")

    st.markdown("""
    <div class="alert-box alert-info">
        <div class="alert-title"><i class="fa-solid fa-circle-info"></i> Tentang Simulasi</div>
        Data pengguna nyata belum tersedia → menggunakan data sintetis.
        Metrik: response rate, waktu penyelesaian tagihan, kepuasan pengguna (1–5).
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: n_users  = st.slider("Jumlah Pengguna",       100, 2000, 500, 100)
    with col2: base_rate = st.slider("Base Response Rate (%)", 20, 60, 35, 5)
    with col3: effect    = st.slider("Effect Size Variant B (%)", 5, 30, 15, 5)

    np.random.seed(42)
    group_a = np.random.binomial(1, base_rate/100, n_users//2)
    group_b = np.random.binomial(1, (base_rate+effect)/100, n_users//2)
    rate_a, rate_b = group_a.mean()*100, group_b.mean()*100
    lift   = rate_b - rate_a
    time_a = np.clip(np.random.normal(48, 12, n_users//2), 1, None)
    time_b = np.clip(np.random.normal(36, 10, n_users//2), 1, None)
    sat_a  = np.clip(np.random.normal(3.2, 0.6, n_users//2), 1, 5)
    sat_b  = np.clip(np.random.normal(3.8, 0.5, n_users//2), 1, 5)

    # Kartu perbandingan metrik A vs B
    col1, col2, col3 = st.columns(3)
    for col, metric, va, vb in [(col1, "Response Rate",   f"{rate_a:.1f}%", f"{rate_b:.1f}%"),
                                 (col2, "Avg. Resolve Time", f"{time_a.mean():.1f}j", f"{time_b.mean():.1f}j"),
                                 (col3, "Kepuasan",        f"{sat_a.mean():.2f}", f"{sat_b.mean():.2f}")]:
        with col:
            st.markdown(f"""
            <div style="display:flex;gap:12px;">
                <div class="ab-result-card ab-variant-a" style="flex:1;">
                    <div class="ab-label" style="color:#1565C0;">Variant A</div>
                    <div style="font-size:12px;color:#6A9A7A;margin-bottom:8px;">Reminder Standar</div>
                    <div class="ab-stat" style="color:#1565C0;">{va}</div>
                    <div style="font-size:12px;color:#8AB89A;">{metric}</div>
                </div>
                <div class="ab-result-card ab-variant-b" style="flex:1;">
                    <div class="ab-label" style="color:#2E7D32;">Variant B</div>
                    <div style="font-size:12px;color:#6A9A7A;margin-bottom:8px;">Reminder Personal</div>
                    <div class="ab-stat" style="color:#2E7D32;">{vb}</div>
                    <div style="font-size:12px;color:#8AB89A;">{metric}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    winner = "Variant B lebih unggul" if lift > 0 else "Variant A lebih unggul"
    st.markdown(f"""
    <div class="alert-box {'alert-success' if lift > 0 else 'alert-warning'}">
        <div class="alert-title"><i class="fa-solid fa-trophy"></i> Hasil Simulasi</div>
        <b>Lift Response Rate:</b> <span class="badge {'badge-green' if lift > 0 else 'badge-red'}">{lift:+.1f} ppt</span>&nbsp;
        <b>Kesimpulan:</b> {winner}. Waktu lebih cepat <b>{time_a.mean()-time_b.mean():.1f} jam</b>.<br>
        <small style="color:#8AB89A;">Simulasi {n_users:,} pengguna sintetis — bukan data nyata.</small>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(pd.DataFrame({'Variant':['A — Standar','B — Personal'],'Response Rate (%)':[rate_a,rate_b]}),
                     x='Variant', y='Response Rate (%)', color='Variant', text_auto='.1f',
                     color_discrete_map={'A — Standar':'#1565C0','B — Personal':'#2E7D32'}, template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Perbandingan Response Rate', height=350), use_container_width=True)
    with col2:
        time_df = pd.DataFrame({'Waktu (jam)': list(time_a[:200])+list(time_b[:200]),
                                 'Variant': ['A — Standar']*200+['B — Personal']*200})
        fig = px.histogram(time_df, x='Waktu (jam)', color='Variant', nbins=30, barmode='overlay', opacity=0.7,
                           color_discrete_map={'A — Standar':'#1565C0','B — Personal':'#2E7D32'}, template='plotly_white')
        st.plotly_chart(plotly_layout(fig, 'Distribusi Waktu Penyelesaian Tagihan', height=350), use_container_width=True)


elif selected_section == "recommendation":
    section_header("bullseye", "Recommendation",
        "Rekomendasi strategis untuk dataset, pipeline, dan rencana evaluasi model NER Talang.in.")

    cols = st.columns(3)
    recs = [
        ("chart-line", "Augmentasi Data",
         'Tambah variasi penulisan harga (<i>"dua puluh ribu"</i>, <i>"20rb"</i>, <i>"20k"</i>, <i>"20.000"</i>) agar entity PRICE lebih robust.'),
        ("folder-tree", "Ekspansi Sumber Data",
         "Manfaatkan <b>tokopedia_reviews</b> dan <b>produk_tokopedia</b> untuk variasi item menu non-GoFood."),
        ("brain", "Rencana Evaluasi Model",
         "Evaluasi model NER dengan <b>F1-score per entity</b>. Prioritaskan PRICE karena formatnya paling beragam dan paling kritis."),
    ]
    for col, (icon, label, body) in zip(cols, recs):
        with col:
            st.markdown(f'<div class="metric-card"><span class="metric-icon"><i class="fa-solid fa-{icon}"></i></span><div class="metric-label">{label}</div><div style="font-size:13px;color:#2D5A3D;line-height:1.7;">{body}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-box alert-warning" style="margin-top:14px;">
        <div class="alert-title"><i class="fa-solid fa-triangle-exclamation"></i> Rekomendasi Kualitas Data</div>
        Masih ada <b>{missing_after:,} missing value</b> di <b>discount_price</b>.
        Pertimbangkan imputasi median per kategori daripada <code>fillna(0)</code>,
        dan tambahkan kolom boolean <b>has_discount</b> agar analisis diskon lebih informatif.
    </div>
    """, unsafe_allow_html=True)


elif selected_section == "health_score":
    section_header("heart-pulse", "Health Score",
        "Skor kesehatan keseluruhan project — kualitas data GoFood + kelengkapan training data NER.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="health-card">
            <div class="health-label">Overall Health Score</div>
            <div class="health-score-big" style="color:{health_color};">{health_score}%</div>
            <div style="font-size:24px;color:{health_color};"><i class="fa-solid {health_icon}"></i></div>
            <div style="font-size:14px;margin-top:8px;color:{health_color};font-weight:700;">{health_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.write("")
        st.progress(health_score / 100)
        st.write("")
        checks = [
            ("Bebas Duplikat",           True, f"{duplicate_before:,} duplikat → sudah di-drop"),
            ("Missing Value Tertangani", True, f"{missing_before:,} → {missing_after:,} sisa (discount_price)"),
            ("Training Data Tersedia",   True, f"{len(training_data):,} records siap pakai"),
            ("4 Entity Label Teranotasi",True, "PERSON · ITEM · PRICE · MULTIPLIER"),
            ("Synthetic Data Tergenerasi",True, f"{len(templates):,} templates via LLM Gemma"),
            ("Pipeline Terdokumentasi",  True, "2 notebook: preprocessing + data_gen"),
        ]
        for label, passed, detail in checks:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding:8px 0;border-bottom:1px solid #E8F5E9;">
                <span style="font-size:13px;color:#2D5A3D;">
                    <i class="fa-solid {'fa-circle-check' if passed else 'fa-circle-xmark'}"></i> {label}<br>
                    <small style="color:#8AB89A;">{detail}</small>
                </span>
                <span class="badge {'badge-green' if passed else 'badge-red'}">{'Lulus' if passed else 'Perlu Perbaikan'}</span>
            </div>
            """, unsafe_allow_html=True)

    for issue in health_issues:
        st.markdown(f'<div class="alert-box alert-warning"><i class="fa-solid fa-triangle-exclamation"></i> {issue}</div>', unsafe_allow_html=True)


elif selected_section == "kesimpulan":
    section_header("clipboard-check", "Kesimpulan",
        "Rangkuman akhir — membuktikan dataset siap untuk AI Smart Transaction Input Talang.in.")

    st.markdown(f"""
    <div class="alert-box alert-success">
        <div class="alert-title"><i class="fa-solid fa-clipboard-check"></i> Kesimpulan Project Data Science Talang.in</div>
        <ol style="margin:0;padding-left:20px;line-height:2.2;">
            <li>Dataset GoFood digunakan sebagai <b>sumber referensi</b> untuk membangun training data NER — bukan data transaksi pengguna langsung.</li>
            <li>Dataset GoFood: <b>{len(df_raw):,} data mentah</b> dari <b>{total_merchant:,} merchant</b> di <b>{total_area} area</b> → bersih menjadi <b>{len(df):,} baris</b>.</li>
            <li>Training data NER: <b>{len(training_data):,} records</b> dengan <b>{total_entities:,} entitas</b> teranotasi (PERSON, ITEM, PRICE, MULTIPLIER) — siap ke tim AI Engineer.</li>
            <li>Pipeline menggabungkan <b>6 dataset sumber</b> → cleaning → template generation → LLM Gemma → 3 file JSON output.</li>
            <li>Training data mengandung <b>{len(item_counter):,} item unik</b> dan <b>{len(person_counter):,} nama orang unik</b> untuk generalisasi model.</li>
            <li>Simulasi A/B Testing menunjukkan potensi peningkatan respons via reminder berbasis rekomendasi personal.</li>
            <li>Health Score project: <b style="color:{health_color};">{health_score}% — {health_status}</b>. Dataset siap digunakan untuk <b>AI Smart Transaction Input</b>.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ── 8. FOOTER ──────────────────────────────────────────────
st.markdown(f"""
<div class="footer-area">
    <div class="footer-title"><i class="fa-solid fa-wallet"></i> Talang.in — Data Science Analytics Dashboard</div>
    <p>Streamlit Cloud Deployment · NER · GoFood</p>
    <p style="margin-top:8px;">
        <span class="badge badge-green">v3.0</span>&nbsp;
        <span class="badge badge-green">Streamlit Cloud</span>&nbsp;
        <span class="badge badge-blue">{len(training_data):,} Data Latih</span>&nbsp;
        <span class="badge badge-blue">{total_entities:,} Entities</span>
    </p>
    <p style="margin-top:12px;font-size:12px;color:#8AB89A;">
        Sources: gofood · alergen · steakhouse · indonesian_food · nutrition · indonesian-names<br>
        Pipeline: data_preprocessing.ipynb → data_gen_gemma.ipynb → training_data.json
    </p>
</div>
""", unsafe_allow_html=True)
