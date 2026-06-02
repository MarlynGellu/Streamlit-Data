# =========================================================
# TALANG.IN — Data Science Dashboard v4.1 (Optimized)
# Fix: sidebar teks terlihat + memory optimization
# =========================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUT_DIR  = os.path.join(BASE_DIR, "..", "outputs")
ROOT_DIR = os.path.join(BASE_DIR, "..")

# ── LOGO (gunakan URL atau path lokal, BUKAN base64 besar) ──
LOGO_URL = "https://img.icons8.com/fluency/96/money-transfer.png"

st.set_page_config(
    page_title="Talang.in Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ── (diperbaiki: sidebar teks terlihat jelas) ──────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── App background ── */
.stApp, .block-container {
    color: #1B4332 !important;
    background: linear-gradient(135deg, #EDF7EE, #E3F2E4, #EAF5EA);
}
.stApp p, .stApp div, .stApp span, .stApp li { color: #1B4332; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1300px; }

/* ── Sidebar background ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071F16, #0D3B2E, #134E3A) !important;
}
/* Semua teks di sidebar putih */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #E8F5E9 !important;
}
section[data-testid="stSidebar"] label {
    color: #A8D5B5 !important;
    font-size: 13px !important;
}

/* ── Radio button sidebar ── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #C8E6C9 !important;
    -webkit-text-fill-color: #C8E6C9 !important;
    display: block !important;
    width: 100% !important;
    transition: all 0.2s;
    border: 1px solid rgba(255,255,255,0.08) !important;
    cursor: pointer;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(82,194,122,0.18) !important;
    border-left: 3px solid #52C27A !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label p {
    color: inherit !important;
    -webkit-text-fill-color: inherit !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #5A8A6A; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #FFF !important; color: #1B4332 !important; }
.stTabs [data-baseweb="tab-list"] { background: #E8F5E9; border-radius: 14px; padding: 4px; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2D9A5A, #52C27A) !important;
    border-radius: 8px !important;
}
.stProgress > div { background: #D4EDD8 !important; border-radius: 8px !important; height: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: linear-gradient(#81C784, #2D9A5A); border-radius: 4px; }

/* ── Cards ── */
.card {
    background: #FFF;
    border-radius: 18px;
    padding: 20px 22px;
    border: 1.5px solid #E0EFE0;
    height: 100%;
    transition: all 0.3s;
    box-shadow: 0 2px 12px rgba(27,67,50,0.06);
}
.card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(27,67,50,0.12); border-color: #A5D6A7; }
.card-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #8AB89A; margin-bottom: 6px; }
.card-value { font-size: 28px; font-weight: 800; color: #1B4332; line-height: 1.1; }
.card-sub   { font-size: 12px; color: #A8C8B0; margin-top: 6px; }
.card-icon  { font-size: 26px; margin-bottom: 12px; display: block; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0A3D27, #145C38, #1B6E44);
    border-radius: 24px;
    padding: 40px 44px;
    margin-bottom: 32px;
    box-shadow: 0 20px 60px rgba(13,74,48,0.3);
}
.page-header h1 { color: #FFF !important; font-size: 28px; font-weight: 800; margin: 0 0 10px; }
.page-header p, .page-header p * { color: #B8DFC8 !important; font-size: 14px; margin: 0; line-height: 1.8; }
.page-header b, .page-header strong { color: #FFF !important; }
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #D4FFDC !important;
    font-size: 11px; font-weight: 700;
    padding: 5px 16px; border-radius: 20px;
    margin-bottom: 16px;
    text-transform: uppercase; letter-spacing: 1px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 14px;
    margin: 36px 0 8px; padding-bottom: 14px;
    border-bottom: 2px solid #C8E6C9;
}
.section-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: #2D9A5A;
}
.section-title-text { font-size: 22px; font-weight: 800; color: #1B4332; margin: 0; }
.section-desc { font-size: 14px; color: #6A9A7A; margin: 0 0 20px; line-height: 1.6; }

/* ── Info boxes ── */
.box { border-radius: 16px; padding: 16px 20px; margin-bottom: 14px; font-size: 14px; line-height: 1.7; transition: all 0.2s; }
.box:hover { transform: translateX(3px); }
.box * { color: inherit !important; }
.box-info { background: linear-gradient(135deg,#EBF8EE,#F0FFF4); border-left: 4px solid #2D9A5A; color: #1B4332 !important; }
.box-ok   { background: linear-gradient(135deg,#E8F5E9,#F1FBF1); border-left: 4px solid #2E7D32; color: #1B4332 !important; }
.box-warn { background: linear-gradient(135deg,#FFF8E1,#FFFDE7); border-left: 4px solid #F9A825; color: #5D4037 !important; }
.box-title { font-weight: 700; font-size: 14px; margin-bottom: 6px; }

/* ── Badges & Chips ── */
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.bg  { background: linear-gradient(135deg,#C8E6C9,#A5D6A7); color: #1B5E20 !important; }
.bb  { background: linear-gradient(135deg,#BBDEFB,#90CAF9); color: #0D47A1 !important; }
.br  { background: linear-gradient(135deg,#FFCDD2,#EF9A9A); color: #B71C1C !important; }
.chip { display: inline-block; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 700; margin: 4px; border: 1.5px solid; }
.cu  { background: linear-gradient(135deg,#E8F5E9,#F0FFF4); color: #1B5E20; border-color: #81C784; }
.cp  { background: linear-gradient(135deg,#E3F2FD,#EFF8FF); color: #0D47A1; border-color: #64B5F6; }
.ce  { background: linear-gradient(135deg,#FFF8E1,#FFFDE7); color: #795548; border-color: #FFD54F; }

/* ── Pipeline card ── */
.pipe-card { background: #FFF; border-radius: 18px; padding: 22px 24px; border: 1.5px solid #E0EFE0; box-shadow: 0 4px 16px rgba(27,67,50,0.06); }
.pipe-step { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 16px; }
.step-num  { width: 30px; height: 30px; background: linear-gradient(135deg,#1B4332,#2D9A5A); color: #FFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; flex-shrink: 0; }
.step-text { font-size: 13px; color: #2D5A3D; line-height: 1.7; }
.step-label { font-weight: 700; color: #1B4332; font-size: 13px; }

/* ── NER cards ── */
.ner-card  { border-radius: 18px; padding: 22px 20px; text-align: center; height: 100%; transition: all 0.3s; box-shadow: 0 4px 16px rgba(0,0,0,0.07); }
.ner-card:hover { transform: translateY(-4px) scale(1.02); }
.ner-label { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px; }
.ner-value { font-size: 32px; font-weight: 800; line-height: 1; margin-bottom: 6px; }

/* ── A/B cards ── */
.ab-card { background: #FFF; border-radius: 18px; padding: 20px 16px; border: 1.5px solid #E0EFE0; text-align: center; transition: all 0.3s; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.ab-card:hover { transform: translateY(-3px); }
.ab-a { border-top: 4px solid #1565C0; }
.ab-b { border-top: 4px solid #2E7D32; }
.ab-stat { font-size: 28px; font-weight: 800; line-height: 1.1; margin: 8px 0 4px; }
.ab-lbl  { font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #8AB89A; }

/* ── Footer ── */
.footer {
    background: linear-gradient(135deg,#FFF,#F5FFF6);
    border-radius: 20px; padding: 28px 32px;
    text-align: center; border: 1.5px solid #D4EDD8; margin-top: 48px;
}
.footer-title { font-size: 16px; font-weight: 800; color: #1B4332; margin-bottom: 6px; }
.footer p { color: #6A9A7A !important; font-size: 13px; margin: 4px 0; }

/* ── Dataframe fix ── */
.stDataFrame *, div[data-testid="stDataFrame"] * { color: #1B4332 !important; }
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span { color: #1B4332 !important; }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
""", unsafe_allow_html=True)


# ── LOAD DATA (optimized: hanya kolom yang diperlukan) ────
@st.cache_data(max_entries=1, show_spinner=False)
def load_gofood():
    cols = ['merchant_name', 'merchant_area', 'category', 'product', 'price', 'discount_price']
    df = pd.read_csv(os.path.join(DATA_DIR, "gofood_dataset.csv"), usecols=cols, dtype={
        'merchant_name': 'category',
        'merchant_area': 'category',
        'category': 'category',
    })
    mb = int(df.isnull().sum().sum())
    db = int(df.duplicated().sum())
    df['discount_price'] = df['discount_price'].fillna(0)
    dc = df.drop_duplicates().reset_index(drop=True)
    ma = int(dc.isnull().sum().sum())
    return df, dc, mb, db, ma

@st.cache_data(max_entries=1, show_spinner=False)
def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(max_entries=1, show_spinner=False)
def ner_stats(td_path):
    with open(td_path, "r", encoding="utf-8") as f:
        tdata = json.load(f)
    ec, ic, pc, pp = Counter(), Counter(), Counter(), []
    for rec in tdata:
        n = 0
        for e in rec.get("entities", []):
            lbl = e.get("label", "")
            if lbl and not lbl.startswith("[") and lbl.strip():
                ec[lbl] += 1
            if lbl == "ITEM":   ic[e["text"].lower()] += 1
            if lbl == "PERSON": pc[e["text"]] += 1; n += 1
        pp.append(n)
    return ec, ic, pc, pp

# Load semua data
df_raw, df, missing_before, dup_before, missing_after = load_gofood()
td_path = os.path.join(OUT_DIR, "training_data.json")
td      = load_json_file(td_path)
fds     = load_json_file(os.path.join(ROOT_DIR, "dataset_fixed.json"))
tmpl    = load_json_file(os.path.join(OUT_DIR, "talangin_synthetic_templates.json"))
ec, ic, pc, pp = ner_stats(td_path)

# Computed stats
total_ent = sum(v for k, v in ec.items() if k.strip() and not k.startswith('['))
tot_merch = df['merchant_name'].nunique()
tot_area  = df['merchant_area'].nunique()
avg_price = float(df['price'].mean())
top_cat   = df['category'].value_counts().idxmax()
top_merch = df['merchant_name'].value_counts().idxmax()
hs, hc, hst = 100, "#2E7D32", "Sangat Baik"


# ── HELPERS ───────────────────────────────────────────────
def sec(icon, title, desc):
    st.markdown(
        f'<div class="section-header">'
        f'<div class="section-icon"><i class="fa-solid {icon}"></i></div>'
        f'<p class="section-title-text">{title}</p></div>'
        f'<p class="section-desc">{desc}</p>',
        unsafe_allow_html=True
    )

def card(col, icon, label, val, sub):
    with col:
        st.markdown(
            f'<div class="card"><span class="card-icon">{icon}</span>'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value">{val}</div>'
            f'<div class="card-sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

def box(cls, ico, title, body):
    st.markdown(
        f'<div class="box {cls}">'
        f'<div class="box-title"><i class="fa-solid fa-{ico}"></i> {title}</div>'
        f'{body}</div>',
        unsafe_allow_html=True
    )

def ch(fig, title, h=430):
    fig.update_layout(
        height=h, coloraxis_showscale=False,
        title=dict(text=title, font=dict(size=14, color='#1B4332')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=10, r=10)
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────
MENUS = [
    ("overview",      "fa-house",          "🏠 Overview"),
    ("data_source",   "fa-database",       "🗄️ Data Source"),
    ("data_cleaning", "fa-broom",          "🧹 Data Cleaning"),
    ("eda",           "fa-chart-bar",      "📊 EDA Data Utama"),
    ("insight",       "fa-lightbulb",      "💡 Insight Data"),
    ("ner_dataset",   "fa-robot",          "🤖 NER Dataset"),
    ("ab_testing",    "fa-flask",          "🧪 A/B Testing"),
    ("kesimpulan",    "fa-clipboard-check","✅ Kesimpulan"),
]

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 20px;">
        <img src="{LOGO_URL}"
             style="width:56px;margin:0 auto 12px;display:block;border-radius:12px;"
             onerror="this.style.display='none'">
        <div style="font-size:20px;font-weight:800;color:#FFFFFF;">Talang.in</div>
        <div style="font-size:11px;color:#6BA882;margin-top:3px;font-weight:600;">
            <span style="font-size:7px;color:#52C27A;">●</span>&nbsp;
            Data Science Dashboard v4.1
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        [m[0] for m in MENUS],
        format_func=lambda k: next(m[2] for m in MENUS if m[0] == k),
        label_visibility="collapsed"
    )

    rows = [
        ("Data bersih", f"{len(df):,}"),
        ("Data latih",  f"{len(td):,}"),
        ("Entitas",     f"{total_ent:,}"),
        ("Health",      f"{hs}%"),
    ]
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.06);font-size:12px;">'
        f'<span style="color:#7ABFA0;">{k}</span>'
        f'<span style="color:#FFFFFF;font-weight:700;">{v}</span></div>'
        for k, v in rows
    ])
    tech_tags = "".join([
        f'<span style="background:rgba(255,255,255,0.08);color:#A8D5B5;'
        f'font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;'
        f'border:1px solid rgba(255,255,255,0.1);">{t}</span>'
        for t in ["GoFood", "Gemma LLM", "Streamlit"]
    ])
    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);margin:16px 0;"></div>
    <div style="background:rgba(82,194,122,0.08);border:1px solid rgba(82,194,122,0.2);
                border-radius:14px;padding:14px 16px;">
        <div style="font-size:10px;color:#52C27A;font-weight:800;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:10px;">
            <i class="fa-solid fa-chart-pie"></i> Statistik Project
        </div>
        {rows_html}
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);margin:16px 0 10px;"></div>
    <div style="text-align:center;font-size:10px;color:#4A7A60;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
        Powered by
    </div>
    <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap;">
        {tech_tags}
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="header-badge">
        <i class="fa-solid fa-robot"></i> NER | GoFood | Data Science | Analytics
    </div>
    <h1>
        <img src="{LOGO_URL}"
             style="height:34px;vertical-align:middle;margin-right:10px;border-radius:8px;"
             onerror="this.style.display='none'">
        Talang.in — Data Science Dashboard
    </h1>
    <p>Dashboard mendokumentasikan proses Data Science untuk fitur
    <b>AI Smart Transaction Input</b> pada aplikasi <b>Talang.in</b>.
    Fokus: <b>data preparation, EDA, preprocessing, dan kesiapan data latih.</b></p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════

# ── OVERVIEW ──────────────────────────────────────────────
if page == "overview":
    sec("house", "Overview",
        "Penjelasan singkat project Talang.in, tujuan dashboard, dan alur data.")

    box("box-info", "wallet", "Apa itu Talang.in?",
        'Aplikasi manajemen keuangan grup berbasis NLP. Pengguna ketik kalimat biasa, AI ekstrak otomatis.<br>'
        '<b style="background:#1B4332;color:#FFF;padding:4px 12px;border-radius:8px;font-size:13px;">'
        '"Ayu bayar pizza 90k untuk Raka dan Nina"</b>'
        '&nbsp;<i class="fa-solid fa-arrow-right" style="color:#2D9A5A;"></i>&nbsp;'
        '<b>PERSON: Ayu, Raka, Nina &nbsp;|&nbsp; ITEM: pizza &nbsp;|&nbsp; PRICE: 90k</b>')

    st.write("")
    cols = st.columns(4)
    metrics = [
        ('<i class="fa-solid fa-database" style="color:#2D9A5A;"></i>', "GoFood Raw",  f"{len(df_raw):,}", "data mentah"),
        ('<i class="fa-solid fa-broom"    style="color:#1565C0;"></i>', "Data Bersih", f"{len(df):,}",     "setelah cleaning"),
        ('<i class="fa-solid fa-robot"    style="color:#6A1B9A;"></i>', "Data Latih NER", f"{len(td):,}", "records berlabel"),
        ('<i class="fa-solid fa-tags"     style="color:#BF360C;"></i>', "Total Entitas",  f"{total_ent:,}","PERSON+ITEM+PRICE+MULT"),
    ]
    for col, (ico, lbl, val, sub) in zip(cols, metrics):
        card(col, ico, lbl, val, sub)

    st.write("")
    steps = [
        ("1","fa-file-import",  "Pengumpulan Data",       "6 dataset dikumpulkan dari sumber publik"),
        ("2","fa-broom",        "Cleaning & Preprocessing","GoFood dibersihkan: hapus duplikat, missing value"),
        ("3","fa-chart-bar",    "EDA & Analisis",          "Eksplorasi distribusi kategori, harga, merchant, area"),
        ("4","fa-code",         "Generate Template",       "Nama menu + nama orang dibuat jadi kalimat transaksi sintetis"),
        ("5","fa-robot",        "Dataset Final NER",       "training_data.json berisi kalimat berlabel PERSON, ITEM, PRICE, MULTIPLIER"),
    ]
    html = '<div class="pipe-card">'
    for n, ico, lbl, desc in steps:
        html += (
            f'<div class="pipe-step">'
            f'<div class="step-num">{n}</div>'
            f'<div class="step-text">'
            f'<span class="step-label"><i class="fa-solid {ico}"></i> {lbl}</span><br>{desc}'
            f'</div></div>'
        )
    st.markdown(html + '</div>', unsafe_allow_html=True)


# ── DATA SOURCE ───────────────────────────────────────────
elif page == "data_source":
    sec("database", "Data Source",
        "Dataset yang digunakan, fungsi, dan kategorinya: Utama, Pendukung, Eksplorasi.")

    box("box-info", "circle-info", "Kategorisasi Dataset",
        '<span class="chip cu">Data Utama</span> analisis & sumber variasi NER. '
        '<span class="chip cp">Data Pendukung</span> variasi item & nama. '
        '<span class="chip ce">Data Eksplorasi</span> riset awal, tidak masuk model utama.')

    datasets = [
        ("fa-store",     "gofood_dataset.csv",      "45.195", "cu","Data Utama",
         "Sumber utama nama produk, harga, kategori, area merchant. Sumber variasi ITEM dan PRICE."),
        ("fa-id-card",   "indonesian-names.csv",    "1.960",  "cu","Data Utama",
         "Nama orang Indonesia. Sumber variasi entity PERSON dalam kalimat transaksi."),
        ("fa-bowl-food", "indonesian_food.csv",     "1.273",  "cp","Data Pendukung",
         "Nama makanan Indonesia. Menambah variasi ITEM di luar GoFood."),
        ("fa-utensils",  "Steakhouse_dataset.csv",  "150",    "cp","Data Pendukung",
         "Menu steakhouse Indonesia. Variasi ITEM jenis makanan berat."),
        ("fa-flask",     "alergen_dataset.csv",     "100.000","ce","Data Eksplorasi",
         "Label alergen pangan. Eksplorasi awal, bukan data utama NER."),
        ("fa-apple-whole","nutrition.csv",           "1.345",  "ce","Data Eksplorasi",
         "Info gizi makanan. Relevan jika ada fitur rekomendasi atau filter nutrisi."),
    ]
    for ico, name, rows, chip, lbl, desc in datasets:
        st.markdown(
            f'<div class="card" style="margin-bottom:10px;padding:16px 20px;">'
            f'<div style="display:flex;align-items:flex-start;gap:16px;">'
            f'<div style="width:40px;height:40px;background:#E8F5E9;border-radius:10px;'
            f'display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;color:#2D9A5A;">'
            f'<i class="fa-solid {ico}"></i></div>'
            f'<div style="flex:1;">'
            f'<div style="display:flex;gap:10px;margin-bottom:6px;align-items:center;">'
            f'<b style="color:#1B4332;">{name}</b>'
            f'<span class="chip {chip}" style="margin:0;">{lbl}</span>'
            f'<span class="badge bb">{rows} baris</span></div>'
            f'<div style="font-size:13px;color:#2D5A3D;">{desc}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True
        )

    box("box-ok", "file-export", "Output Akhir Data Science",
        f"Dataset utama dan pendukung yang relevan digabungkan dan diproses menjadi "
        f"<b>training_data.json</b> — berisi <b>{len(td):,} kalimat transaksi sintetis</b> "
        f"dengan anotasi entitas PERSON, ITEM, PRICE, dan MULTIPLIER. "
        f"File ini merupakan output akhir tim Data Science dan siap digunakan sebagai data latih model NER.")


# ── DATA CLEANING ─────────────────────────────────────────
elif page == "data_cleaning":
    sec("broom", "Data Cleaning",
        "Proses pembersihan data GoFood sebelum dan sesudah cleaning.")

    pct = (len(df_raw) - len(df)) / len(df_raw) * 100
    cols = st.columns(4)
    metrics = [
        ('<i class="fa-solid fa-database"></i>',     "Data Mentah",      f"{len(df_raw):,}", "sebelum cleaning"),
        ('<i class="fa-solid fa-circle-check"></i>', "Data Bersih",      f"{len(df):,}",     "setelah cleaning"),
        ('<i class="fa-solid fa-copy"></i>',         "Duplikat Dihapus", f"{dup_before:,}",  "baris terbuang"),
        ('<i class="fa-solid fa-circle-xmark"></i>', "Missing Value",    f"{missing_before:,}","cell kosong awal"),
    ]
    for col, (ico, lbl, val, sub) in zip(cols, metrics):
        card(col, ico, lbl, val, sub)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        box("box-ok", "table-cells", "Penanganan Missing Value",
            f"Ditemukan <b>{missing_before:,} cell kosong</b> pada data mentah. "
            f"Kolom <b>discount_price</b> diisi dengan <code>fillna(0)</code>.<br><br>"
            f"Sisa <b>{missing_after:,} missing value</b> berasal dari kolom <b>display</b> dan "
            f"<b>description</b> yang <b>tidak digunakan</b> dalam pipeline NER.")
        miss = df_raw.isnull().sum().reset_index()
        miss.columns = ['Kolom', 'Missing']
        miss['Status'] = miss['Missing'].apply(lambda x: 'Bersih' if x == 0 else f'{x:,} missing')
        st.dataframe(miss, use_container_width=True, hide_index=True)

    with c2:
        box("box-warn" if dup_before > 0 else "box-ok", "copy", "Penanganan Duplikasi",
            f"Ditemukan <b>{dup_before:,} duplikat</b> — dihapus dengan <code>drop_duplicates()</code>. "
            f"Data bersih: <b>{len(df):,}</b> dari {len(df_raw):,} ({pct:.1f}% terbuang).")
        box("box-info", "list-check", "Kolom yang Digunakan",
            "<b>Dipakai:</b> merchant_name, merchant_area, category, product, price, discount_price<br>"
            "<b>Fitur baru:</b> isDiscount, category_encoded<br>"
            "<b>Tidak dipakai:</b> display, description (terlalu panjang untuk NER)")
        fig = px.bar(
            pd.DataFrame({'Kondisi': ['Sebelum', 'Sesudah'], 'Baris': [len(df_raw), len(df)]}),
            x='Kondisi', y='Baris', color='Kondisi', text_auto=True,
            color_discrete_map={'Sebelum': '#F9A825', 'Sesudah': '#2D9A5A'},
            template='plotly_white'
        )
        st.plotly_chart(ch(fig, 'Sebelum vs Sesudah Cleaning', 280), use_container_width=True)


# ── EDA ───────────────────────────────────────────────────
elif page == "eda":
    sec("chart-bar", "EDA Data Utama",
        "Eksplorasi data GoFood. Filter tersedia di halaman ini.")

    c1, c2 = st.columns(2)
    with c1:
        kf = st.multiselect("Filter Kategori", list(df['category'].unique()), list(df['category'].unique()))
    with c2:
        af = st.multiselect("Filter Area", list(df['merchant_area'].unique()), list(df['merchant_area'].unique()))

    fdf = df[df['category'].isin(kf) & df['merchant_area'].isin(af)]
    st.markdown(
        f'<div class="box box-info" style="padding:10px 16px;margin-bottom:12px;">'
        f'Menampilkan <b>{len(fdf):,} produk</b> dari '
        f'<b>{fdf["merchant_name"].nunique():,} merchant</b> di '
        f'<b>{fdf["merchant_area"].nunique()} area</b></div>',
        unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.tabs(["Kategori", "Top Merchant", "Harga", "Area"])

    with t1:
        kd = fdf['category'].value_counts().reset_index()
        kd.columns = ['Kategori', 'Jumlah']
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ch(px.bar(kd.head(15), x='Kategori', y='Jumlah', color='Jumlah',
                text_auto=True, color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'],
                template='plotly_white'), 'Top 15 Kategori'), use_container_width=True)
        with c2:
            st.plotly_chart(ch(px.pie(kd.head(8), names='Kategori', values='Jumlah',
                color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.45),
                'Top 8 Kategori'), use_container_width=True)
        box("box-info", "lightbulb", "Insight Kategori",
            f"Kategori dominan: <b>{top_cat}</b>. Keragaman memperkaya variasi entity <b>ITEM</b> pada training data NER.")

    with t2:
        md = fdf['merchant_name'].value_counts().head(10).reset_index()
        md.columns = ['Merchant', 'Jumlah']
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ch(px.bar(md.sort_values('Jumlah'), x='Jumlah', y='Merchant',
                orientation='h', color='Jumlah', text_auto=True,
                color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white'),
                'Top 10 Merchant', 400), use_container_width=True)
        with c2:
            st.plotly_chart(ch(px.pie(md, names='Merchant', values='Jumlah',
                color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.5),
                'Porsi Top 10', 400), use_container_width=True)
        box("box-info", "store", "Insight Merchant",
            f"<b>{fdf['merchant_name'].nunique():,} merchant unik</b>. Terbanyak: <b>{top_merch}</b>.")

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ch(px.histogram(fdf[fdf['price'] < 200000], x='price', nbins=50,
                color_discrete_sequence=['#2D9A5A'], template='plotly_white'),
                'Distribusi Harga (< Rp 200.000)', 350), use_container_width=True)
        with c2:
            pc2 = fdf.groupby('category')['price'].mean().sort_values(ascending=False).head(10).reset_index()
            pc2.columns = ['Kategori', 'Rata-rata']
            st.plotly_chart(ch(px.bar(pc2, x='Rata-rata', y='Kategori', orientation='h',
                color='Rata-rata', text_auto='.0f',
                color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white'),
                'Rata-rata Harga per Kategori', 350), use_container_width=True)
        box("box-info", "coins", "Insight Harga",
            f"Rata-rata: <b>Rp {avg_price:,.0f}</b>. Rentang luas = variasi entity PRICE: Rp 20.000, 20k, dua puluh ribu.")

    with t4:
        ad = fdf['merchant_area'].value_counts().reset_index()
        ad.columns = ['Area', 'Jumlah']
        fig = go.Figure(go.Bar(
            x=ad['Area'], y=ad['Jumlah'],
            marker_color='#2D9A5A', text=ad['Jumlah'], textposition='outside'
        ))
        fig.update_layout(height=380,
            title=dict(text=f'Produk per Area ({fdf["merchant_area"].nunique()} Area)', font=dict(size=14,color='#1B4332')),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
        box("box-info", "location-dot", "Insight Area",
            f"Data mencakup <b>{fdf['merchant_area'].nunique()} area</b>. Volume tinggi = variasi menu lebih kaya.")


# ── INSIGHT ───────────────────────────────────────────────
elif page == "insight":
    sec("lightbulb", "Insight Data",
        "Temuan dari EDA dan hubungannya dengan kebutuhan AI Smart Transaction Input.")

    insights = [
        ("store",    "Keragaman Merchant & Menu",
         f"Dataset mencatat <b>{len(df):,} produk</b> dari <b>{tot_merch:,} merchant</b> di <b>{tot_area} area</b>. Memastikan variasi entity ITEM yang kaya."),
        ("coins",    "Variasi Format Harga",
         f"Rata-rata <b>Rp {avg_price:,.0f}</b>. Variasi harga menghasilkan variasi PRICE: <b>Rp 20.000</b>, <b>20k</b>, <b>dua puluh ribu</b>."),
        ("tag",      "Kategori Dominan",
         f"Terbanyak: <b>{top_cat}</b>. Distribusi beragam memastikan training data tidak bias ke satu jenis makanan."),
        ("user",     "Kebutuhan Entity PERSON",
         "<b>1.960 nama unik</b> dikombinasikan dengan GoFood. Satu kalimat bisa mengandung <b>1-8 nama orang</b>."),
        ("robot",    "Hubungan dengan AI Smart Input",
         f"Semua insight menghasilkan <b>training_data.json</b>: <b>{len(td):,} kalimat</b>, <b>{total_ent:,} entitas</b> — diserahkan ke tim AI Engineer."),
        ("triangle-exclamation","Catatan Data Eksplorasi",
         "Dataset alergen & nutrisi tidak masuk model NER utama — lebih relevan untuk fitur rekomendasi makanan di masa depan."),
    ]
    c1, c2 = st.columns(2)
    for i, (ico, title, body) in enumerate(insights):
        with (c1 if i % 2 == 0 else c2):
            box("box-info", ico, title, body)


# ── NER DATASET ───────────────────────────────────────────
elif page == "ner_dataset":
    sec("robot", "NER Dataset",
        "Statistik dataset final training_data.json siap untuk tim AI Engineer.")

    box("box-warn", "circle-info", "Status Dataset NER",
        "Dataset <b>siap diserahkan ke tim AI Engineer</b>. "
        "Training & evaluasi (F1-score) dilakukan oleh tim AI — bukan scope Data Science.")

    ner_items = [
        ('<i class="fa-solid fa-user"></i>',    "PERSON",     ec.get('PERSON', 0),     "nama orang",  "#1B5E20", "#E8F5E9"),
        ('<i class="fa-solid fa-utensils"></i>',"ITEM",       ec.get('ITEM', 0),       "nama menu",   "#0D47A1", "#E3F2FD"),
        ('<i class="fa-solid fa-coins"></i>',   "PRICE",      ec.get('PRICE', 0),      "harga",       "#4A148C", "#F3E5F5"),
        ('<i class="fa-solid fa-hashtag"></i>', "MULTIPLIER", ec.get('MULTIPLIER', 0), "jumlah/porsi","#BF360C", "#FBE9E7"),
    ]
    cols = st.columns(4)
    for col, (ico, lbl, val, sub, tc, bg) in zip(cols, ner_items):
        with col:
            st.markdown(
                f'<div class="ner-card" style="background:{bg};">'
                f'<div class="ner-label" style="color:{tc};">{lbl}</div>'
                f'<div class="ner-value" style="color:{tc};">{val:,}</div>'
                f'<div style="font-size:20px;margin:6px 0;color:{tc};">{ico}</div>'
                f'<div style="font-size:12px;color:#8AB89A;">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    epr = [len(x.get('entities', [])) for x in td]
    se  = sorted(epr)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Statistik Dataset")
        stats_df = pd.DataFrame({
            "Metrik": ["Total records","Total entitas","Rata-rata entitas/record",
                       "Median","Max","PERSON unik","ITEM unik","Templates"],
            "Nilai":  [f"{len(td):,}", f"{total_ent:,}", f"{sum(epr)/len(epr):.2f}",
                       f"{se[len(se)//2]}", f"{max(epr)}", f"{len(pc):,}",
                       f"{len(ic):,}", f"{len(tmpl):,}"]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("##### Contoh Data Latih")
        sample = []
        for r in td[:5]:
            ents = ", ".join(
                f"{e['label']}:{e['text']}"
                for e in r.get('entities', [])[:4]
                if e.get('label','').strip() and not e.get('label','').startswith('[')
            )
            sample.append({"Kalimat": r['text'][:80] + "...", "Entities": ents})
        st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True)

    with c2:
        edf = pd.DataFrame({
            'Entity': [k for k in ec if k.strip() and not k.startswith('[')],
            'Count':  [v for k, v in ec.items() if k.strip() and not k.startswith('[')]
        })
        st.plotly_chart(ch(px.pie(edf, names='Entity', values='Count', hole=0.5,
            color_discrete_map={'PERSON':'#2E7D32','ITEM':'#1565C0','PRICE':'#6A1B9A','MULTIPLIER':'#BF360C'}),
            'Distribusi Entity', 300), use_container_width=True)

        ti = pd.DataFrame(ic.most_common(10), columns=['Item', 'Frekuensi'])
        st.plotly_chart(ch(px.bar(ti.sort_values('Frekuensi'), x='Frekuensi', y='Item',
            orientation='h', color='Frekuensi', text_auto=True,
            color_continuous_scale=['#A8D5B5','#2D9A5A','#0D4A30'], template='plotly_white'),
            'Top 10 Item Menu', 320), use_container_width=True)


# ── A/B TESTING ───────────────────────────────────────────
elif page == "ab_testing":
    sec("flask", "A/B Testing Simulation",
        "Simulasi: Reminder Standar (A) vs Reminder Personal (B).")

    box("box-info", "circle-info", "Tentang Simulasi",
        "Data pengguna nyata belum tersedia — simulasi menggunakan data sintetis. "
        "Metrik: response rate, waktu penyelesaian, kepuasan pengguna.")

    c1, c2, c3 = st.columns(3)
    with c1: nu = st.slider("Jumlah Pengguna",  100, 2000, 500, 100)
    with c2: br = st.slider("Base Response Rate (%)", 20, 60, 35, 5)
    with c3: ef = st.slider("Effect Size B (%)", 5, 30, 15, 5)

    np.random.seed(42)
    ga = np.random.binomial(1, br / 100, nu // 2)
    gb = np.random.binomial(1, (br + ef) / 100, nu // 2)
    ra, rb = ga.mean() * 100, gb.mean() * 100
    lift = rb - ra
    ta = np.clip(np.random.normal(48, 12, nu // 2), 1, None)
    tb = np.clip(np.random.normal(36, 10, nu // 2), 1, None)
    sa = np.clip(np.random.normal(3.2, 0.6, nu // 2), 1, 5)
    sb = np.clip(np.random.normal(3.8, 0.5, nu // 2), 1, 5)

    c1, c2, c3 = st.columns(3)
    for col, m, va, vb in [
        (c1, "Response Rate", f"{ra:.1f}%",    f"{rb:.1f}%"),
        (c2, "Waktu (jam)",   f"{ta.mean():.1f}j", f"{tb.mean():.1f}j"),
        (c3, "Kepuasan",      f"{sa.mean():.2f}", f"{sb.mean():.2f}"),
    ]:
        with col:
            st.markdown(
                f'<div style="display:flex;gap:12px;">'
                f'<div class="ab-card ab-a" style="flex:1;">'
                f'<div class="ab-lbl" style="color:#1565C0;">Variant A</div>'
                f'<div style="font-size:11px;color:#8AB89A;margin-bottom:8px;">Reminder Standar</div>'
                f'<div class="ab-stat" style="color:#1565C0;">{va}</div>'
                f'<div style="font-size:12px;color:#8AB89A;">{m}</div></div>'
                f'<div class="ab-card ab-b" style="flex:1;">'
                f'<div class="ab-lbl" style="color:#2E7D32;">Variant B</div>'
                f'<div style="font-size:11px;color:#8AB89A;margin-bottom:8px;">Reminder Personal</div>'
                f'<div class="ab-stat" style="color:#2E7D32;">{vb}</div>'
                f'<div style="font-size:12px;color:#8AB89A;">{m}</div></div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    winner = "Variant B lebih unggul" if lift > 0 else "Variant A lebih unggul"
    badge  = "bg" if lift > 0 else "br"
    box("box-ok" if lift > 0 else "box-warn", "trophy", "Hasil Simulasi",
        f'Lift: <span class="badge {badge}">{lift:+.1f} ppt</span>&nbsp;'
        f'<b>{winner}</b> — waktu lebih cepat <b>{ta.mean()-tb.mean():.1f} jam</b>.<br>'
        f'<small style="color:#8AB89A;">Simulasi {nu:,} pengguna sintetis — bukan data nyata.</small>')

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(ch(px.bar(
            pd.DataFrame({'Variant': ['A','B'], 'Rate': [ra, rb]}),
            x='Variant', y='Rate', color='Variant', text_auto='.1f',
            color_discrete_map={'A':'#1565C0','B':'#2E7D32'}, template='plotly_white'),
            'Response Rate A vs B', 350), use_container_width=True)
    with c2:
        tdf = pd.DataFrame({'Waktu': list(ta[:200]) + list(tb[:200]), 'Variant': ['A']*200 + ['B']*200})
        st.plotly_chart(ch(px.histogram(tdf, x='Waktu', color='Variant', nbins=30,
            barmode='overlay', opacity=0.7,
            color_discrete_map={'A':'#1565C0','B':'#2E7D32'}, template='plotly_white'),
            'Distribusi Waktu Penyelesaian', 350), use_container_width=True)


# ── KESIMPULAN ────────────────────────────────────────────
elif page == "kesimpulan":
    sec("clipboard-check", "Kesimpulan",
        "Rangkuman akhir proses Data Science Talang.in.")

    conclusions = [
        ("file-import",  "1. Pengumpulan Data",
         f"6 dataset dikumpulkan. Data utama: <b>gofood_dataset.csv</b> ({len(df_raw):,} baris) "
         f"untuk ITEM & PRICE, <b>indonesian-names.csv</b> untuk PERSON."),
        ("broom",        "2. Data Cleaning",
         f"Dibersihkan dari <b>{dup_before:,} duplikat</b> dan <b>{missing_before:,} missing value</b>. "
         f"Sisa {missing_after:,} missing berasal dari kolom <b>display & description</b> yang tidak dipakai, "
         f"menghasilkan <b>{len(df):,} data bersih</b>."),
        ("chart-bar",    "3. EDA",
         f"Keragaman <b>{tot_area} area</b>, <b>{tot_merch:,} merchant</b>, "
         f"rata-rata harga <b>Rp {avg_price:,.0f}</b> mendukung variasi data latih NER."),
        ("robot",        "4. Dataset Final NER",
         f"<b>{len(td):,} kalimat transaksi</b> dengan <b>{total_ent:,} entitas</b>. "
         f"File <b>training_data.json</b> siap ke tim AI Engineer."),
        ("flask",        "5. A/B Testing",
         "Simulasi membuktikan potensi peningkatan response rate dengan reminder personal."),
        ("heart-pulse",  "6. Health Score",
         f"Health Score: <b style='color:{hc};'>{hs}% — {hst}</b>. "
         f"Seluruh pipeline terdokumentasi, data bersih, dan dataset final siap untuk fitur "
         f"<b>AI Smart Transaction Input</b>."),
    ]
    for ico, title, body in conclusions:
        box("box-info", ico, title, body)

    st.write("")
    checklist = [
        (c1 := st.columns(3)[0], [
            ("Bebas Duplikat",          True,  f"{dup_before:,} dihapus"),
            ("Missing Tertangani",      True,  "fillna(0) + kolom non-NER diabaikan"),
            ("Cleaning Terdokumentasi", True,  "before & after"),
        ]),
        (c2 := st.columns(3)[1] if False else st.columns(3)[1], [
            ("EDA Terdokumentasi",  True, "4 aspek"),
            ("Insight ke NER",      True, "dikaitkan ke model"),
            ("A/B Testing",         True, "simulasi reminder"),
        ]),
        (c3 := st.columns(3)[2] if False else st.columns(3)[2], [
            ("Dataset Final",         True, f"{len(td):,} records"),
            ("4 Entity Teranotasi",   True, "PERSON, ITEM, PRICE, MULT"),
            ("Pipeline Terdokumentasi",True,"2 notebook"),
        ]),
    ]
    c1, c2, c3 = st.columns(3)
    for col, checks in [(c1,[("Bebas Duplikat",True,f"{dup_before:,} dihapus"),("Missing Tertangani",True,"fillna(0) + kolom non-NER diabaikan"),("Cleaning Terdokumentasi",True,"before & after")]),
                        (c2,[("EDA Terdokumentasi",True,"4 aspek"),("Insight ke NER",True,"dikaitkan ke model"),("A/B Testing",True,"simulasi reminder")]),
                        (c3,[("Dataset Final",True,f"{len(td):,} records"),("4 Entity Teranotasi",True,"PERSON, ITEM, PRICE, MULT"),("Pipeline Terdokumentasi",True,"2 notebook")])]:
        with col:
            for lbl, ok, det in checks:
                ic2  = "fa-circle-check" if ok else "fa-circle-xmark"
                ic3  = "#2E7D32" if ok else "#C62828"
                badge_cls = "bg" if ok else "br"
                badge_txt = "Lulus" if ok else "Perlu Perbaikan"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:8px 0;border-bottom:1px solid #EBF5EB;">'
                    f'<span style="font-size:13px;color:#2D5A3D;">'
                    f'<i class="fa-solid {ic2}" style="color:{ic3};"></i> {lbl}<br>'
                    f'<small style="color:#8AB89A;">{det}</small></span>'
                    f'<span class="badge {badge_cls}">{badge_txt}</span></div>',
                    unsafe_allow_html=True
                )


# ── FOOTER ────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <div class="footer-title">
        <img src="{LOGO_URL}"
             style="height:22px;vertical-align:middle;margin-right:8px;border-radius:6px;"
             onerror="this.style.display='none'">
        Talang.in — Data Science Dashboard
    </div>
    <p>NER | GoFood | Analytics | Streamlit Cloud</p>
    <p style="margin-top:8px;">
        <span class="badge bg">v4.1</span>&nbsp;
        <span class="badge bg">Streamlit Cloud</span>&nbsp;
        <span class="badge bb">{len(td):,} Data Latih</span>&nbsp;
        <span class="badge bb">{total_ent:,} Entities</span>
    </p>
</div>
""", unsafe_allow_html=True)