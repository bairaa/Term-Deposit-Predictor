import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(
    page_title="PortuBank — Term Deposit Predictor",
    page_icon="portubank_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

  html, body, [class*="css"] {
    font-family: 'Spectral', Georgia, serif !important;
  }
  [data-testid="stSidebarNav"] *,
  .material-icons, [class*="material-icon"],
  button svg, [data-testid] svg {
    font-family: inherit !important;
  }

  .stApp { background-color: #f5f0e8; }
  .block-container { padding-top: 1.5rem !important; }

  section[data-testid="stSidebar"] {
    background-color: #faf7f2;
    border-right: 1px solid #ddd5c0;
  }

  .header-banner {
    background: linear-gradient(135deg, #1a3a5c 0%, #2a5480 50%, #1a3a5c 100%);
    border: 1px solid #c9a84c66;
    border-radius: 12px;
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, #c9a84c22, transparent 65%);
    pointer-events: none;
  }
  .header-banner h1 {
    font-family: 'Spectral', serif !important;
    font-size: 1.85rem;
    font-weight: 600;
    color: #f0e6cc;
    margin: 0 0 0.3rem 0;
  }
  .header-banner p {
    color: #a8bdd4;
    margin: 0;
    font-size: 0.9rem;
    font-weight: 300;
  }
  .bank-tag {
    display: inline-block;
    background: #c9a84c22;
    border: 1px solid #c9a84c88;
    color: #c9a84c;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 3px;
    margin-bottom: 0.75rem;
  }

  .metric-card {
    background: #ffffff;
    border: 1px solid #ddd5c0;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px #0001;
  }
  .metric-card .label {
    color: #8a7a60;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
  }
  .metric-card .value {
    font-size: 1.55rem;
    font-weight: 600;
    color: #1a3a5c;
  }
  .metric-card .sub {
    color: #b0a080;
    font-size: 0.72rem;
    margin-top: 0.15rem;
  }

  /* Result cards */
  .result-positive {
    background: linear-gradient(135deg, #edfaf3, #d4f5e3);
    border: 1.5px solid #2da86a;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    text-align: center;
    box-shadow: 0 2px 12px #2da86a18;
  }
  .result-negative {
    background: linear-gradient(135deg, #fef2f2, #fde4e4);
    border: 1.5px solid #e05252;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    text-align: center;
    box-shadow: 0 2px 12px #e0525218;
  }
  .result-title {
    font-family: 'Spectral', serif !important;
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0.4rem 0 0.2rem;
  }
  .result-positive .result-title { color: #1a6b40; }
  .result-negative .result-title { color: #b83232; }
  .result-prob {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0.3rem 0;
  }
  .result-positive .result-prob { color: #1d8a50; }
  .result-negative .result-prob { color: #c94040; }

  /* Priority tier cards */
.tier-1 {
    background: #fecaca;
    border: 1.5px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 0.9rem;
    box-shadow: 0 2px 8px #ef444420;
  }
  .tier-2 {
    background: #fed7aa;
    border: 1.5px solid #f97316;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 0.9rem;
    box-shadow: 0 2px 8px #f9731620;
  }
  .tier-3 {
    background: #e5e7eb;
    border: 1.5px solid #9ca3af;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 0.9rem;
    box-shadow: 0 2px 8px #9ca3af20;
  }
  .tier-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }
  .tier-1 .tier-label { color: #991b1b; }
  .tier-2 .tier-label { color: #9a3412; }
  .tier-3 .tier-label { color: #4b5563; }
  .tier-title {
    font-family: 'Spectral', serif !important;
    font-size: 1.1rem;
    font-weight: 600;
  }
  .tier-1 .tier-title { color: #7f1d1d; }
  .tier-2 .tier-title { color: #7c2d12; }
  .tier-3 .tier-title { color: #374151; }
  .tier-desc {
    font-size: 0.8rem;
    margin-top: 0.2rem;
    line-height: 1.5;
  }
  .tier-1 .tier-desc { color: #991b1b; }
  .tier-2 .tier-desc { color: #9a3412; }
  .tier-3 .tier-desc { color: #4b5563; }

  /* ROI box */
  .roi-box {
    background: #ffffff;
    border: 1px solid #ddd5c0;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 1rem;
    box-shadow: 0 1px 4px #0001;
  }
  .roi-box .roi-label {
    color: #8a6820;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .section-title {
    font-family: 'Spectral', serif !important;
    color: #1a3a5c;
    font-size: 1rem;
    font-weight: 600;
    border-bottom: 1.5px solid #ddd5c0;
    padding-bottom: 0.35rem;
    margin-bottom: 0.9rem;
  }
  .sidebar-section {
    font-family: 'Spectral', serif !important;
    color: #1a3a5c;
    font-size: 0.88rem;
    font-weight: 600;
    border-bottom: 1px solid #ddd5c0;
    padding-bottom: 0.3rem;
    margin: 1rem 0 0.7rem 0;
  }

  .stSelectbox label, .stNumberInput label,
  .stSlider label, .stRadio label, .stRadio > label {
    color: #4a3f2f !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
  }
  div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #c8b99a !important;
  }
  input[type="number"] {
    background-color: #ffffff !important;
    color: #1a3a5c !important;
  }
  .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #c9a84c, #2da86a) !important;
  }

  hr { border-color: #ddd5c0 !important; }

  .info-card {
    background: #ffffff;
    border: 1px solid #ddd5c0;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    font-size: 0.83rem;
    box-shadow: 0 1px 4px #0001;
  }

  .footer {
    text-align: center;
    color: #b0a080;
    font-size: 0.73rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #ddd5c0;
  }
</style>
""", unsafe_allow_html=True)

# Constants
COST_PER_CALL    = 1.09
REVENUE_PER_CUST = 37.50

# Load Artifacts
@st.cache_resource
def load_model():
    try:
        with open("bank_telemarketing_model_f6.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

artifacts = load_model()

# Helper: Priority Tier Logic
def get_priority_tier(prob, artifacts):
    """
    Kembalikan (tier_label, priority_config) berdasarkan probabilitas.
    Hitung manual dari priority_config — tidak bergantung pada fungsi
    assign_tier yang di-pickle (menghindari AttributeError saat load).
    """
    if artifacts is None:
        return None, {}

    cfg       = artifacts.get("priority_config", {})
    threshold = cfg.get("threshold", artifacts.get("best_threshold", 0.5))
    q_hot     = cfg.get("q_hot",  threshold + 0.2)
    q_warm    = cfg.get("q_warm", threshold + 0.1)

    if prob >= q_hot:
        return "Tingkat 1 \u2014 Tinggi", cfg
    elif prob >= q_warm:
        return "Tingkat 2 \u2014 Sedang", cfg
    elif prob >= threshold:
        return "Tingkat 3 \u2014 Rendah", cfg
    else:
        return "Tidak Dihubungi", cfg


# Header
st.markdown("""
<div class="header-banner">
  <div style="display:flex;align-items:center;gap:1.2rem">
    <img src="app/static/portubank_logo.png" style="width:64px;height:64px;object-fit:contain;filter:drop-shadow(0 2px 6px #0004)">
    <div>
      <div class="bank-tag">PortuBank</div>
      <h1>Term Deposit Subscription Predictor</h1>
      <p>Sistem Prediksi Berbasis Machine Learning untuk Meminimalkan Biaya Loss Kampanye Telemarketing Deposito Berjangka</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Metric Cards
if artifacts:
    meta       = artifacts.get("metadata", {})
    model_name = meta.get("model_name", "LightGBM")
    f6_test    = meta.get("f6_test", None)
    roc_auc    = meta.get("roc_auc_test", None)
    threshold  = artifacts.get("best_threshold",
                    artifacts.get("priority_config", {}).get("threshold", 0.5))
    cfg        = artifacts.get("priority_config", {})
    q_hot      = cfg.get("q_hot", "—")
    q_warm     = cfg.get("q_warm", "—")
else:
    model_name, f6_test, roc_auc, threshold, q_hot, q_warm = "—", None, None, "—", "—", "—"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="label">Model</div>
      <div class="value" style="font-size:0.9rem;padding-top:0.5rem">{model_name}</div>
      <div class="sub">+ Under-sampling</div>
    </div>""", unsafe_allow_html=True)
with c2:
    f6_disp = f"{f6_test:.4f}" if f6_test else "—"
    st.markdown(f"""<div class="metric-card">
      <div class="label">F6 Score</div>
      <div class="value">{f6_disp}</div>
      <div class="sub">Test Set</div>
    </div>""", unsafe_allow_html=True)
with c3:
    auc_disp = f"{roc_auc:.4f}" if roc_auc else "—"
    st.markdown(f"""<div class="metric-card">
      <div class="label">ROC-AUC</div>
      <div class="value">{auc_disp}</div>
      <div class="sub">Test Set</div>
    </div>""", unsafe_allow_html=True)
with c4:
    thr_disp = f"{threshold:.4f}" if isinstance(threshold, float) else str(threshold)
    st.markdown(f"""<div class="metric-card">
      <div class="label">Threshold F6</div>
      <div class="value">{thr_disp}</div>
      <div class="sub">Cut-off model</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
      <div class="label">Biaya FN / FP</div>
      <div class="value" style="font-size:1.1rem;padding-top:0.3rem">€37.50 / €1.09</div>
      <div class="sub">Rasio FN:FP ≈ 34×</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not artifacts:
    st.warning("⚠️ File `bank_telemarketing_model_f6.pkl` tidak ditemukan. Letakkan di folder yang sama dengan `app.py`, lalu restart.")

# Sidebar Input
with st.sidebar:
    st.markdown("""<div style="text-align:center;margin-bottom:1rem;padding-bottom:0.8rem;border-bottom:1px solid #ddd5c0">
      <span style="font-family:'Spectral',serif;font-size:1.15rem;font-weight:600;color:#1a3a5c">Data Nasabah</span><br>
      <span style="font-size:0.75rem;color:#8a7a60;font-style:italic">Prediksi & Priority Tier diperbarui otomatis</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">👤 Demografi</div>', unsafe_allow_html=True)
    age       = st.number_input("Usia (tahun)", min_value=17, max_value=99, value=35)
    job       = st.selectbox("Pekerjaan", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                             "management", "retired", "self-employed", "services",
                             "student", "technician", "unemployed", "unknown"])
    marital   = st.selectbox("Status Pernikahan", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Pendidikan", ["basic.4y", "basic.6y", "basic.9y", "high.school",
                             "illiterate", "professional.course", "university.degree", "unknown"])

    st.markdown('<div class="sidebar-section">💳 Profil Keuangan</div>', unsafe_allow_html=True)
    default = st.selectbox("Kredit Macet", ["no", "yes", "unknown"])
    housing = st.selectbox("Pinjaman Rumah", ["yes", "no", "unknown"])
    loan    = st.selectbox("Pinjaman Pribadi", ["no", "yes", "unknown"])

    st.markdown('<div class="sidebar-section">📞 Kampanye Saat Ini</div>', unsafe_allow_html=True)
    contact     = st.selectbox("Tipe Kontak", ["cellular", "telephone"])
    month       = st.selectbox("Bulan Kontak", ["jan","feb","mar","apr","may","jun",
                               "jul","aug","sep","oct","nov","dec"], index=4)
    day_of_week = st.selectbox("Hari Kontak", ["mon","tue","wed","thu","fri"])
    campaign    = st.number_input("Jumlah Kontak (kampanye ini)", min_value=1, max_value=56, value=1)

    st.markdown('<div class="sidebar-section">📋 Riwayat Kampanye</div>', unsafe_allow_html=True)
    pdays_contacted = st.radio("Pernah dihubungi sebelumnya?", ["Belum pernah", "Pernah"], horizontal=True)
    pdays    = 999 if pdays_contacted == "Belum pernah" else st.number_input("Hari sejak kontak terakhir", 0, 999, 30)
    previous = st.number_input("Jumlah kontak sebelum kampanye ini", 0, 40, 0)
    poutcome = st.selectbox("Hasil kampanye sebelumnya", ["nonexistent", "failure", "success"])

    st.markdown('<div class="sidebar-section">📊 Ekonomi Makro</div>', unsafe_allow_html=True)
    emp_var_rate   = st.number_input("Employment Variation Rate", -3.4, 1.4, -1.8, step=0.1, format="%.1f")
    cons_price_idx = st.number_input("Consumer Price Index", 92.0, 95.0, 93.2, step=0.1, format="%.3f")
    cons_conf_idx  = st.number_input("Consumer Confidence Index", -51.0, -26.0, -42.0, step=0.1, format="%.1f")
    euribor3m      = st.number_input("Euribor 3M Rate (%)", 0.6, 5.1, 1.3, step=0.01, format="%.3f")
    nr_employed    = st.number_input("Nr. Employed (ribuan)", 4963.0, 5228.0, 5099.0, step=1.0, format="%.1f")

# Feature Engineering (sesuai pipeline notebook)
def build_input():
    contacted_before = 0 if pdays_contacted == "Belum pernah" else 1
    age_group        = str(pd.cut([age], bins=[17, 30, 40, 50, 60, 100],
                                  labels=["18-30","31-40","41-50","51-60","60+"])[0])
    euribor_level    = str(pd.cut([euribor3m], bins=[0, 1, 2, 3, 4, 5.1],
                                  labels=["very_low","low","medium","high","very_high"])[0])
    season_map       = {"mar":"Q1_peak","apr":"Q2","may":"Q2","jun":"Q2",
                        "jul":"Q3","aug":"Q3","sep":"Q3_peak","oct":"Q4_peak",
                        "nov":"Q4","dec":"Q4_peak"}
    month_season     = season_map.get(month, "Q2")
    campaign_intensity = str(pd.cut([min(campaign, 20)], bins=[0, 1, 3, 6, 20],
                                    labels=["once","few","many","excessive"])[0])

    job_score_map = {"management":3,"entrepreneur":3,"self-employed":2,"admin.":2,
                     "technician":2,"retired":2,"services":1,"housemaid":1,
                     "blue-collar":1,"student":0,"unemployed":0,"unknown":1}
    edu_score_map = {"university.degree":2,"professional.course":2,"high.school":1,
                     "basic.9y":1,"basic.6y":0,"basic.4y":0,"illiterate":0,"unknown":1}
    def age_score(a):
        return 1 if a < 25 else 2 if a < 35 else 3 if a < 55 else 2

    wealth_proxy = job_score_map.get(job, 1) + edu_score_map.get(education, 1) + age_score(age)

    return pd.DataFrame([{
        "campaign": campaign, "previous": previous,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx, "nr.employed": nr_employed,
        "wealth_proxy_score": wealth_proxy,
        "contacted_before": contacted_before,
        "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan,
        "contact": contact, "day_of_week": day_of_week, "poutcome": poutcome,
        "age_group": age_group, "euribor_level": euribor_level,
        "month_season": month_season, "campaign_intensity": campaign_intensity,
    }])


# Main Layout
main_col, info_col = st.columns([3, 2], gap="large")

with main_col:
    if not artifacts:
        st.markdown("""<div style="background:#fff;border:1.5px dashed #c8b99a;border-radius:12px;
                    padding:3rem 2rem;text-align:center">
          <div style="font-size:2.5rem;margin-bottom:1rem">🏦</div>
          <div style="font-family:'Spectral',serif;color:#8a7a60;font-size:1.05rem">
            Model belum tersedia.<br>Pastikan file <code>bank_telemarketing_model_f6.pkl</code>
            sudah ditempatkan di folder yang sama dengan <code>app.py</code>.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        try:
            pipeline     = artifacts["pipeline"]
            preprocessor = artifacts["preprocessor"]
            threshold    = artifacts.get("best_threshold",
                               artifacts.get("priority_config", {}).get("threshold", 0.5))

            X_processed = preprocessor.transform(build_input())
            prob        = pipeline.predict_proba(X_processed)[0][1]
            decision    = prob >= threshold
            exp_return  = prob * REVENUE_PER_CUST - COST_PER_CALL

            # Prediksi Utama
            if decision:
                st.markdown(f"""
                <div class="result-positive">
                  <div style="font-size:2rem">✅</div>
                  <div class="result-title">LAYAK DIHUBUNGI</div>
                  <div class="result-prob">{prob*100:.1f}%</div>
                  <div style="color:#1a6b40;font-size:0.85rem">
                    Probabilitas subscribe melampaui threshold F6 ({threshold:.4f})
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                  <div style="font-size:2rem">❌</div>
                  <div class="result-title">TIDAK DIREKOMENDASIKAN</div>
                  <div class="result-prob">{prob*100:.1f}%</div>
                  <div style="color:#b83232;font-size:0.85rem">
                    Probabilitas subscribe di bawah threshold F6 ({threshold:.4f})
                  </div>
                </div>""", unsafe_allow_html=True)

            # Priority Tier (hanya tampil jika layak dihubungi)
            tier_label, tier_cfg = get_priority_tier(prob, artifacts)

            if decision and tier_label:
                if "Tingkat 1" in tier_label:
                    tier_class = "tier-1"
                    tier_icon  = "🔴"
                    tier_action = "Hubungi <strong>pertama</strong> — alokasikan ke agen senior dengan closing rate terbaik."
                    tier_budget = "Prioritas saat anggaran terbatas (33% kapasitas)."
                elif "Tingkat 2" in tier_label:
                    tier_class = "tier-2"
                    tier_icon  = "🟡"
                    tier_action = "Hubungi <strong>setelah Tingkat 1</strong> — peluang moderat, efektif saat kapasitas cukup."
                    tier_budget = "Prioritas saat anggaran sedang (~67% kapasitas)."
                else:
                    tier_class = "tier-3"
                    tier_icon  = "⚪"
                    tier_action = "Hubungi <strong>terakhir</strong> — masuk daftar kontak, tapi eksekusi saat kapasitas tersedia."
                    tier_budget = "Hubungi jika anggaran penuh (100% kapasitas)."

                q_hot_val  = tier_cfg.get("q_hot",  "—")
                q_warm_val = tier_cfg.get("q_warm", "—")
                q_hot_str  = f"{q_hot_val:.4f}" if isinstance(q_hot_val, float) else str(q_hot_val)
                q_warm_str = f"{q_warm_val:.4f}" if isinstance(q_warm_val, float) else str(q_warm_val)

                st.markdown(f"""
                <div class="{tier_class}">
                  <div class="tier-label">{tier_icon} Priority Tier</div>
                  <div class="tier-title">{tier_label}</div>
                  <div class="tier-desc">
                    {tier_action}<br>
                    <span style="font-size:0.76rem;opacity:0.8">{tier_budget}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            # ROI Box
            st.markdown(f"""
            <div class="roi-box">
              <div class="roi-label">📈 Analisis Ekonomi Panggilan</div>
              <table style="width:100%;margin-top:0.6rem;border-collapse:collapse">
                <tr>
                  <td style="color:#6b5a40;font-size:0.83rem;padding:0.3rem 0">Probabilitas Subscribe</td>
                  <td style="color:#1a3a5c;font-size:0.83rem;text-align:right;font-weight:600">{prob*100:.2f}%</td>
                </tr>
                <tr>
                  <td style="color:#6b5a40;font-size:0.83rem;padding:0.3rem 0">Threshold F6-Optimal</td>
                  <td style="color:#8a6820;font-size:0.83rem;text-align:right;font-weight:600">{threshold:.4f}</td>
                </tr>
                <tr>
                  <td style="color:#6b5a40;font-size:0.83rem;padding:0.3rem 0">Expected Return per Panggilan</td>
                  <td style="color:{'#1a6b40' if exp_return >= 0 else '#b83232'};font-size:0.83rem;text-align:right;font-weight:700">
                    {'+'if exp_return>=0 else ''}€{exp_return:.2f}
                  </td>
                </tr>
                <tr>
                  <td style="color:#6b5a40;font-size:0.83rem;padding:0.3rem 0">Biaya per Panggilan (FP)</td>
                  <td style="color:#1a3a5c;font-size:0.83rem;text-align:right">€{COST_PER_CALL:.2f}</td>
                </tr>
                <tr>
                  <td style="color:#6b5a40;font-size:0.83rem;padding:0.3rem 0">Revenue jika Subscribe (FN cost)</td>
                  <td style="color:#1a3a5c;font-size:0.83rem;text-align:right">€{REVENUE_PER_CUST:.2f}</td>
                </tr>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="color:#6b5a40;font-size:0.78rem;margin-bottom:0.3rem">'
                f'Probabilitas Subscribe: <b style="color:#1a3a5c">{prob*100:.1f}%</b></div>',
                unsafe_allow_html=True
            )
            st.progress(min(prob, 1.0))

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")


with info_col:
    # Tentang Model
    st.markdown('<div class="section-title">ℹ️ Tentang Model & Threshold</div>', unsafe_allow_html=True)
    thr_disp2 = f"{threshold:.4f}" if isinstance(threshold, float) else str(threshold)
    st.markdown(f"""<div class="info-card">
      <div style="color:#4a3f2f;line-height:1.7;margin-bottom:0.8rem">
        Model menggunakan <b style="color:#8a6820">F6-optimal threshold</b> —
        bukan 0.5 default — karena biaya 1 FN (€37,50 revenue NIM hilang)
        <b>34× lebih mahal</b> dari biaya 1 FP (€1,09 per panggilan).
      </div>
      <div style="border-top:1px solid #ddd5c0;padding-top:0.8rem">
        <div style="color:#8a7a60;font-size:0.7rem;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:0.5rem">Threshold yang digunakan</div>
        <code style="background:#f5f0e8;color:#8a6820;padding:0.4rem 0.7rem;border-radius:6px;display:block;font-size:0.85rem;border:1px solid #ddd5c0">
          F6-optimal threshold = <b>{thr_disp2}</b>
        </code>
      </div>
    </div>""", unsafe_allow_html=True)

    # Priority Tier Guide
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Panduan Priority Tier</div>', unsafe_allow_html=True)

    cfg_disp   = artifacts.get("priority_config", {}) if artifacts else {}
    q_hot_d    = cfg_disp.get("q_hot",  "—")
    q_warm_d   = cfg_disp.get("q_warm", "—")
    thr_d      = cfg_disp.get("threshold", threshold)
    q_hot_s    = f"{q_hot_d:.4f}" if isinstance(q_hot_d, float) else str(q_hot_d)
    q_warm_s   = f"{q_warm_d:.4f}" if isinstance(q_warm_d, float) else str(q_warm_d)
    thr_s      = f"{thr_d:.4f}" if isinstance(thr_d, float) else str(thr_d)

    for icon, label, prob_range, desc, color in [
        ("🔴", "Tingkat 1 — Tinggi", f"prob ≥ {q_hot_s}",
         "Top 33% dari predicted-yes. Agen senior, dihubungi pertama.", "#e07b00"),
        ("🟡", "Tingkat 2 — Sedang", f"{q_warm_s} ≤ prob < {q_hot_s}",
         "Mid 33%. Dihubungi setelah Tingkat 1 selesai.", "#b89000"),
        ("⚪", "Tingkat 3 — Rendah", f"{thr_s} ≤ prob < {q_warm_s}",
         "Bottom 33% predicted-yes. Dihubungi saat kapasitas penuh.", "#888"),
        ("—",  "Tidak Dihubungi",    f"prob < {thr_s}",
         "Tidak masuk daftar kontak kampanye.", "#ccc"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:0.7rem;margin-bottom:0.85rem">
          <div style="width:3px;min-height:52px;background:{color};border-radius:2px;flex-shrink:0;margin-top:3px"></div>
          <div>
            <div style="color:#1a3a5c;font-size:0.83rem;font-weight:600">{icon} {label}</div>
            <div style="color:#8a6820;font-size:0.72rem;font-family:monospace">{prob_range}</div>
            <div style="color:#8a7a60;font-size:0.75rem">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Prediktor Kunci
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Prediktor Kunci</div>', unsafe_allow_html=True)
    for feat, desc, color in [
        ("poutcome = success",         "~65% conversion rate",         "#2da86a"),
        ("Euribor3m rendah (<2%)",     "Kondisi ideal kampanye",       "#2563eb"),
        ("Pernah dihubungi sebelumnya","Nasabah 'warm lead'",          "#7c3aed"),
        ("Bulan: Mar, Sep, Oct, Dec",  "Timing optimal kampanye",      "#c9a84c"),
        ("Contact: cellular",          "Lebih efektif vs telephone",   "#0891b2"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:0.7rem;margin-bottom:0.75rem">
          <div style="width:3px;min-height:38px;background:{color};border-radius:2px;flex-shrink:0;margin-top:3px"></div>
          <div>
            <div style="color:#1a3a5c;font-size:0.83rem;font-weight:600">{feat}</div>
            <div style="color:#8a7a60;font-size:0.75rem">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Cara Baca Hasil
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Cara Membaca Hasil</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card" style="font-size:0.82rem;line-height:1.7">
      <div style="margin-bottom:0.6rem">
        <span style="color:#1a6b40;font-weight:600">✅ Layak Dihubungi</span><br>
        <span style="color:#4a3f2f">Expected return positif. Probabilitas melampaui threshold F6-optimal.
        Lihat Priority Tier untuk menentukan urutan kontak.</span>
      </div>
      <div style="margin-bottom:0.6rem">
        <span style="color:#e07b00;font-weight:600">🔴 Tingkat 1 — Tinggi</span><br>
        <span style="color:#4a3f2f">Probabilitas tertinggi (top 33%). Hubungi pertama, terutama
        saat anggaran kampanye terbatas.</span>
      </div>
      <div>
        <span style="color:#b83232;font-weight:600">❌ Tidak Direkomendasikan</span><br>
        <span style="color:#4a3f2f">Expected return negatif. Biaya panggilan melebihi
        ekspektasi pendapatan NIM dari nasabah ini.</span>
      </div>
    </div>""", unsafe_allow_html=True)


# Authors
AUTHORS = [
    {
        "name"      : "M. Hafizh Hariyanto",
        "role"      : "Data Scientist",
        "linkedin"  : "https://www.linkedin.com/in/muhammad-hafizh-hariyanto/",
        "portfolio" : "https://hafizhhariyanto9.github.io/",
    },
    {
        "name"      : "Baira Rahayu",
        "role"      : "Data Scientist",
        "linkedin"  : "https://www.linkedin.com/in/baira-rahayu/",
        "portfolio" : "https://bairaa.github.io/bairasportfolio/",
    },
]

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title" style="max-width:900px;margin:0 auto 1rem">👥 Tim Pengembang</div>', unsafe_allow_html=True)

author_cols = st.columns(len(AUTHORS))
for col, author in zip(author_cols, AUTHORS):
    with col:
        linkedin_btn  = f'<a href="{author["linkedin"]}" target="_blank" style="text-decoration:none"><div style="display:inline-flex;align-items:center;gap:0.4rem;background:#0a66c2;color:#fff;font-size:0.75rem;font-weight:600;padding:0.35rem 0.8rem;border-radius:5px;margin-right:0.4rem">🔗 LinkedIn</div></a>' if author["linkedin"] else ""
        portfolio_btn = f'<a href="{author["portfolio"]}" target="_blank" style="text-decoration:none"><div style="display:inline-flex;align-items:center;gap:0.4rem;background:#1a3a5c;color:#fff;font-size:0.75rem;font-weight:600;padding:0.35rem 0.8rem;border-radius:5px">🌐 Portfolio</div></a>' if author["portfolio"] else ""
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #ddd5c0;border-radius:10px;
                    padding:1.2rem 1.4rem;text-align:center;box-shadow:0 1px 4px #0001">
          <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#1a3a5c,#2a5480);
                      display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem;
                      font-size:1.3rem;color:#f0e6cc;font-weight:700">
            {author["name"][0].upper()}
          </div>
          <div style="font-family:'Spectral',serif;font-size:1rem;font-weight:600;color:#1a3a5c;margin-bottom:0.2rem">
            {author["name"]}
          </div>
          <div style="font-size:0.75rem;color:#8a7a60;margin-bottom:0.9rem">{author["role"]}</div>
          <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.4rem">
            {linkedin_btn}{portfolio_btn}
          </div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
  PortuBank · Term Deposit Telemarketing Optimization System ·
  Model: LightGBM F6-optimized · Dataset: UCI Bank Marketing (Moro et al., 2014)
</div>
""", unsafe_allow_html=True)
