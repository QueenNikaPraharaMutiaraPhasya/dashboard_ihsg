import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path("artifacts")
st.set_page_config(
    page_title="IHSG Analytics | Prediksi ARIMA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# PALET WARNA
# ------------------------------------------------------------
NAVY, ROYAL, SKY, TINT = "#0F4C81", "#1D4ED8", "#0284C7", "#E0F2FE"
INK, SLATE, MUTED, LINE = "#0F172A", "#334155", "#64748B", "#E2E8F0"
BG, WHITE = "#F8FAFC", "#FFFFFF"
HIJAU, MERAH = "#16A34A", "#DC2626"

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# Kompatibilitas versi Streamlit (width="stretch" vs use_container_width)
try:
    _v = st.__version__.split(".")
    WIDE = {"width": "stretch"} if (int(_v[0]), int(_v[1])) >= (1, 50) else {"use_container_width": True}
except Exception:
    WIDE = {"use_container_width": True}

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp { background: #F8FAFC; color: #0F172A; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
.side-brand { font-size: 18px; font-weight: 800; color: #0F4C81; letter-spacing: -0.3px; }
.side-brand span { color: #0284C7; }
.side-sub { font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.side-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px; margin-top: 10px; }
.side-row { display: flex; justify-content: space-between; gap: 10px; font-size: 12px; padding: 5px 0; border-bottom: 1px dashed #E2E8F0; }
.side-row:last-child { border-bottom: none; }
.side-row .k { color: #64748B; }
.side-row .v { color: #0F172A; font-weight: 600; text-align: right; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0F4C81 0%, #1D4ED8 62%, #0284C7 100%);
    border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(15, 76, 129, 0.22);
}
.hero, .hero * { color: #FFFFFF !important; }
.hero-top { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
.hero-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 2px; opacity: 0.85; margin-bottom: 6px; }
.hero-title { font-size: 30px; font-weight: 800; letter-spacing: -0.6px; line-height: 1.15; }
.hero-sub { font-size: 13.5px; opacity: 0.9; margin-top: 6px; max-width: 640px; }
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    display: inline-block; padding: 6px 13px; border-radius: 999px; font-size: 12px; font-weight: 600;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.30);
}
.hero .badge-solid { background: #FFFFFF; color: #0F4C81 !important; border-color: #FFFFFF; }

/* KPI */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 8px; }
.kpi-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 14px 0; }
.kpi-grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 14px 0; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } .kpi-grid-4 { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 700px)  { .kpi-grid, .kpi-grid-3, .kpi-grid-4 { grid-template-columns: repeat(2, 1fr); } }
.kpi {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px;
    border-top: 3px solid #0F4C81; transition: all .18s ease;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.kpi:hover { transform: translateY(-2px); border-color: #93C5FD; box-shadow: 0 8px 20px rgba(29, 78, 216, 0.12); }
.kpi-label { font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 1.1px; }
.kpi-value { font-size: 26px; font-weight: 800; color: #0F172A; letter-spacing: -0.6px; margin-top: 6px; line-height: 1.1; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 12px; font-weight: 600; margin-top: 6px; color: #64748B; }
.kpi-sub.up { color: #16A34A; }
.kpi-sub.down { color: #DC2626; }
.kpi-sub.info { color: #0284C7; }

/* Section */
.sec-title { font-size: 18px; font-weight: 700; color: #0F172A; margin: 20px 0 2px 0; letter-spacing: -0.2px; }
.sec-title:before { content: ""; display: inline-block; width: 4px; height: 16px; background: #1D4ED8; border-radius: 2px; margin-right: 10px; vertical-align: -2px; }
.sec-desc { font-size: 13px; color: #64748B; margin-bottom: 10px; }
.note { font-size: 12.5px; color: #334155; background: #E0F2FE; border-left: 3px solid #0284C7; border-radius: 8px; padding: 9px 13px; margin: 8px 0 4px 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 18px; border-radius: 10px 10px 0 0; color: #64748B; font-weight: 600; background: transparent; }
.stTabs [data-baseweb="tab"]:hover { color: #0F4C81; background: #F1F5F9; }
.stTabs [aria-selected="true"] { color: #0F4C81 !important; background: #E0F2FE; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #0F4C81; height: 3px; }

/* Chart & tabel */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 6px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; }

/* Kartu & tabel spesifikasi */
.card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; overflow: hidden; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
.spec { width: 100%; border-collapse: collapse; font-size: 13px; }
.spec th { background: #0F4C81; color: #FFFFFF; text-align: left; font-weight: 600; padding: 10px 14px; font-size: 12px; letter-spacing: .5px; }
.spec td { padding: 10px 14px; border-bottom: 1px solid #EEF2F7; color: #334155; vertical-align: top; }
.spec tr:last-child td { border-bottom: none; }
.spec tr:hover td { background: #F8FAFC; }
.spec td.k { font-weight: 600; color: #0F172A; white-space: nowrap; }
.spec td.v { font-weight: 700; color: #0F4C81; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 700; }
.pill.ok { background: #DCFCE7; color: #166534; }
.pill.bad { background: #FEE2E2; color: #991B1B; }
.pill.blue { background: #E0F2FE; color: #075985; }

/* Tombol unduh */
.stDownloadButton button {
    background: #0F4C81; color: #FFFFFF; border: none; border-radius: 10px; font-weight: 600; padding: 8px 18px;
    transition: all .15s ease;
}
.stDownloadButton button p { color: #FFFFFF; }
.stDownloadButton button:hover { background: #1D4ED8; color: #FFFFFF; transform: translateY(-1px); box-shadow: 0 6px 14px rgba(29, 78, 216, 0.25); }

.foot { text-align: center; font-size: 11.5px; color: #64748B; margin-top: 28px; padding-top: 14px; border-top: 1px solid #E2E8F0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------
def html(s, target=st):
    """Render HTML tanpa indentasi (agar tidak dianggap code block oleh Markdown)."""
    target.markdown("\n".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def idn(v, d=2):
    """Format angka Indonesia: 1.234,56"""
    s = f"{v:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def idn_signed(v, d=2):
    return ("+" if v >= 0 else "−") + idn(abs(v), d)


def fmt_tgl(d):
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def kpi_card(label, value, sub="", tone="", hint=""):
    sub_html = f'<div class="kpi-sub {tone}">{sub}</div>' if sub else ""
    return (f'<div class="kpi" title="{hint}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{sub_html}</div>')


def section(title, desc=""):
    html(f'<div class="sec-title">{title}</div>' + (f'<div class="sec-desc">{desc}</div>' if desc else ""))


def style_fig(fig, height=420, ytitle=None, ytick=",.0f", unified=True):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=SLATE),
        margin=dict(l=12, r=16, t=44, b=12),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(bgcolor=WHITE, bordercolor=LINE, font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        separators=",.",  # desimal koma, ribuan titik
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor=LINE, tickfont=dict(color=MUTED),
        hoverformat="%d %b %Y",
        showspikes=unified, spikemode="across", spikethickness=1, spikecolor="#94A3B8", spikedash="dot",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EEF2F7", gridwidth=1, zeroline=False,
        tickfont=dict(color=MUTED), tickformat=ytick, title_text=ytitle,
        title_font=dict(color=MUTED, size=12),
    )
    return fig


def kategori_mape(v):
    if v < 10:
        return "Sangat Akurat"
    if v <= 20:
        return "Akurat"
    if v <= 50:
        return "Cukup Akurat"
    return "Tidak Akurat"


# ------------------------------------------------------------
# LOAD DATA & MODEL
# ------------------------------------------------------------
@st.cache_data
def load_data():
    pred = pd.read_csv(ART / "hasil_prediksi.csv", parse_dates=["Tanggal"])
    full = pd.read_csv(ART / "data_bersih.csv", parse_dates=["Tanggal"]).sort_values("Tanggal")
    evaluasi = pd.read_csv(ART / "tabel_evaluasi.csv")
    gs_aic = pd.read_csv(ART / "grid_search_aic.csv")
    with open(ART / "info_model.json") as f:
        info_model = json.load(f)
    with open(ART / "info_split.json") as f:
        split = json.load(f)

    # Arah pergerakan harian dibandingkan penutupan hari sebelumnya
    full["Prev"] = full["Terakhir"].shift(1)
    pred = pred.merge(full[["Tanggal", "Prev"]], on="Tanggal", how="left")
    pred["Prev"] = pred["Prev"].fillna(pred["Aktual"])
    pred["Aktual_Chg%"] = (pred["Aktual"] / pred["Prev"] - 1) * 100
    pred["Prediksi_Chg%"] = (pred["Prediksi"] / pred["Prev"] - 1) * 100
    pred["Arah_Aktual"] = np.where(pred["Aktual"] >= pred["Prev"], "Naik", "Turun")
    pred["Arah_Prediksi"] = np.where(pred["Prediksi"] >= pred["Prev"], "Naik", "Turun")
    pred["Arah_Tepat"] = pred["Arah_Aktual"] == pred["Arah_Prediksi"]
    return pred, full, evaluasi, gs_aic, info_model, split


@st.cache_resource
def load_model():
    return joblib.load(ART / "model_arima.joblib")


pred, full, evaluasi, gs_aic, info_model, split = load_data()
order_str = f"ARIMA({info_model['p']}, {info_model['d']}, {info_model['q']})"

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
html('<div class="side-brand">IHSG<span>.</span>Analytics</div>'
     '<div class="side-sub">ARIMA Forecast Terminal</div>', st.sidebar)

tmin, tmax = pred["Tanggal"].min().date(), pred["Tanggal"].max().date()
rentang = st.sidebar.date_input("Rentang tanggal (data uji)", (tmin, tmax), min_value=tmin, max_value=tmax)
if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
    d1, d2 = rentang
else:
    d1, d2 = tmin, tmax
p = pred[(pred["Tanggal"].dt.date >= d1) & (pred["Tanggal"].dt.date <= d2)].copy()

html(f"""
<div class="side-card">
<div class="side-row"><span class="k">Model</span><span class="v">{order_str}</span></div>
<div class="side-row"><span class="k">Data latih</span><span class="v">{fmt_tgl(pd.to_datetime(split['train_start']).date())} – {fmt_tgl(pd.to_datetime(split['train_end']).date())}</span></div>
<div class="side-row"><span class="k">Jumlah latih</span><span class="v">{idn(split['n_train'], 0)} hari</span></div>
<div class="side-row"><span class="k">Data uji</span><span class="v">{fmt_tgl(pd.to_datetime(split['test_start']).date())} – {fmt_tgl(pd.to_datetime(split['test_end']).date())}</span></div>
<div class="side-row"><span class="k">Jumlah uji</span><span class="v">{idn(split['n_test'], 0)} hari</span></div>
</div>
""", st.sidebar)

# ------------------------------------------------------------
# HEADER (EXECUTIVE SUMMARY)
# ------------------------------------------------------------
html(f"""
<div class="hero">
<div class="hero-top">
<div>
<div class="hero-eyebrow">IDX COMPOSITE · EXECUTIVE SUMMARY</div>
<div class="hero-title">Prediksi Harian IHSG</div>
<div class="hero-sub">Prediksi harga penutupan IHSG di masa pemerintahan Prabowo–Gibran menggunakan {order_str}.</div>
</div>
<div class="hero-badges">
<span class="badge badge-solid">{order_str}</span>
<span class="badge">Periode aktif: {fmt_tgl(d1)} – {fmt_tgl(d2)}</span>
<span class="badge">{idn(len(p), 0)} hari bursa</span>
</div>
</div>
</div>
""")

if p.empty:
    st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    st.stop()

# ------------------------------------------------------------
# KPI
# ------------------------------------------------------------
mae = p["Selisih"].abs().mean()
rmse = np.sqrt((p["Selisih"] ** 2).mean())
mape = (p["Selisih"].abs() / p["Aktual"]).mean() * 100
akurasi_arah = p["Arah_Tepat"].mean() * 100
chg_last = p["Aktual_Chg%"].iloc[-1]

html('<div class="kpi-grid">' + "".join([
    kpi_card("MAE", idn(mae), "Rata-rata galat absolut (poin)", "info", "Mean Absolute Error"),
    kpi_card("RMSE", idn(rmse), "Galat kuadratik (poin)", "info", "Root Mean Squared Error"),
    kpi_card("MAPE", f"{idn(mape)}%", kategori_mape(mape), "info", "Mean Absolute Percentage Error"),
    kpi_card("Akurasi Arah", f"{idn(akurasi_arah, 1)}%", f"{int(p['Arah_Tepat'].sum())} dari {len(p)} hari tepat", "info",
             "Ketepatan prediksi naik/turun"),
    kpi_card("Penutupan Terakhir", idn(p["Aktual"].iloc[-1]),
             f"{'▲' if chg_last >= 0 else '▼'} {idn_signed(chg_last)}% vs hari sebelumnya",
             "up" if chg_last >= 0 else "down", "Harga penutupan aktual pada akhir periode terpilih"),
]) + "</div>")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Aktual vs Prediksi", "Naik / Turun Harian", "Evaluasi & Model", "Forecast Ke Depan"]
)

# ------------------------------------------------------------
# TAB 1: AKTUAL VS PREDIKSI
# ------------------------------------------------------------
with tab1:
    section(f"Harga Penutupan IHSG: Aktual vs Prediksi {order_str}",
            "Perbandingan harga penutupan aktual dengan prediksi satu langkah ke depan pada periode terpilih.")
    tampil_latih = st.checkbox("Tampilkan data latih (riwayat sebelum periode uji)", value=False)

    fig = go.Figure()
    if tampil_latih:
        latih = full[full["Tanggal"] < pd.to_datetime(split["test_start"])]
        fig.add_trace(go.Scatter(x=latih["Tanggal"], y=latih["Terakhir"], name="Data latih",
                                 line=dict(color="#94A3B8", width=1.4), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=p["Tanggal"], y=p["Aktual"], name="Aktual",
                             line=dict(color=NAVY, width=2.6), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=p["Tanggal"], y=p["Prediksi"], name=f"Prediksi {order_str}",
                             line=dict(color=SKY, width=2.4, dash="dash"), hovertemplate="%{y:,.2f}"))
    style_fig(fig, height=450, ytitle="IHSG")
    st.plotly_chart(fig, key="chart_aktual_prediksi", **WIDE)

    section("Selisih Prediksi (Aktual − Prediksi)")
    warna = np.where(p["Selisih"] >= 0, HIJAU, MERAH)
    fig_e = go.Figure(go.Bar(x=p["Tanggal"], y=p["Selisih"], marker_color=warna, name="Selisih",
                             hovertemplate="%{y:+,.2f} poin"))
    fig_e.add_hline(y=0, line_color=SLATE, line_width=1.2)
    style_fig(fig_e, height=300, ytitle="Poin")
    fig_e.update_layout(showlegend=False, margin=dict(l=12, r=16, t=16, b=12))
    st.plotly_chart(fig_e, key="chart_selisih", **WIDE)
    html('<div class="note"><b>Hijau</b>: model memprediksi lebih rendah dari aktual (under-predict). '
         '<b>Merah</b>: model memprediksi lebih tinggi dari aktual (over-predict).</div>')

# ------------------------------------------------------------
# TAB 2: NAIK / TURUN HARIAN
# ------------------------------------------------------------
with tab2:
    section("Perubahan Harian Prediksi vs Aktual (%)",
            "Perubahan dihitung terhadap harga penutupan hari bursa sebelumnya.")

    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(x=p["Tanggal"], y=p["Prediksi_Chg%"], name="Prediksi",
                           marker_color=np.where(p["Prediksi_Chg%"] >= 0, HIJAU, MERAH),
                           opacity=0.85, hovertemplate="%{y:+,.2f}%"))
    fig_c.add_trace(go.Scatter(x=p["Tanggal"], y=p["Aktual_Chg%"], name="Aktual",
                               mode="lines+markers", line=dict(color=NAVY, width=1.8),
                               marker=dict(size=4, color=NAVY), hovertemplate="%{y:+,.2f}%"))
    fig_c.add_hline(y=0, line_color=SLATE, line_width=1.2)
    style_fig(fig_c, height=430, ytitle="Perubahan (%)", ytick=".2f")
    st.plotly_chart(fig_c, key="chart_perubahan", **WIDE)

    naik_p = int((p["Arah_Prediksi"] == "Naik").sum())
    turun_p = int((p["Arah_Prediksi"] == "Turun").sum())
    naik_a = int((p["Arah_Aktual"] == "Naik").sum())
    turun_a = int((p["Arah_Aktual"] == "Turun").sum())

    a, b = st.columns([1, 1.2])
    with a:
        section("Ringkasan Arah")
        st.dataframe(pd.DataFrame({"Prediksi": [naik_p, turun_p], "Aktual": [naik_a, turun_a]},
                                  index=["Naik", "Turun"]), **WIDE)
        html('<div class="kpi-grid-3" style="grid-template-columns:1fr;">' + kpi_card(
            "Arah Tepat", f"{int(p['Arah_Tepat'].sum())} / {len(p)} hari",
            f"{idn(akurasi_arah, 1)}% akurasi arah", "info") + "</div>")
    with b:
        section("Confusion Matrix Arah", "Baris: aktual · Kolom: prediksi")
        cm = pd.crosstab(p["Arah_Aktual"], p["Arah_Prediksi"]).reindex(
            index=["Naik", "Turun"], columns=["Naik", "Turun"], fill_value=0)
        fig_cm = go.Figure(go.Heatmap(
            z=cm.values, x=["Prediksi Naik", "Prediksi Turun"], y=["Aktual Naik", "Aktual Turun"],
            colorscale=[[0, "#E0F2FE"], [1, "#7DB9F0"]], showscale=False,
            texttemplate="%{z}", textfont=dict(size=22, color=INK),
            hovertemplate="%{y} / %{x}: %{z} hari<extra></extra>", xgap=4, ygap=4,
        ))
        style_fig(fig_cm, height=240, unified=False)
        fig_cm.update_layout(margin=dict(l=12, r=12, t=12, b=12), showlegend=False)
        fig_cm.update_yaxes(autorange="reversed", showgrid=False, tickformat=None)
        fig_cm.update_xaxes(hoverformat=None)
        st.plotly_chart(fig_cm, key="chart_cm", **WIDE)

    section("Detail Harian", "Tabel mengikuti rentang tanggal pada filter sidebar.")
    detail = p[["Tanggal", "Prev", "Aktual", "Prediksi", "Aktual_Chg%", "Prediksi_Chg%",
                "Arah_Aktual", "Arah_Prediksi", "Arah_Tepat"]].copy()
    detail["Tanggal"] = detail["Tanggal"].dt.date
    detail = detail.rename(columns={"Prev": "Penutupan Sebelumnya"})
    st.dataframe(detail.round(2), height=330, hide_index=True, **WIDE)
    st.download_button(
        "⬇  Unduh detail harian (CSV)",
        data=detail.round(4).to_csv(index=False).encode("utf-8"),
        file_name=f"detail_harian_ihsg_{d1}_{d2}.csv",
        mime="text/csv",
        key="dl_detail",
    )

# ------------------------------------------------------------
# TAB 3: EVALUASI & MODEL ARIMA
# ------------------------------------------------------------
with tab3:
    lb_ok = info_model["lb_pvalue"] > 0.05
    status_lb = "White Noise (Memenuhi Asumsi)" if lb_ok else "Belum White Noise"

    section("Spesifikasi & Diagnostik Model", "Ringkasan parameter model terpilih dan uji kelayakan residual.")
    html('<div class="kpi-grid-4">' + "".join([
        kpi_card("Model", order_str, f"p={info_model['p']} · d={info_model['d']} · q={info_model['q']}", "info"),
        kpi_card("AIC", idn(info_model["aic"], 2), "Makin kecil makin baik", "info"),
        kpi_card("BIC", idn(info_model["bic"], 2), "Penalti kompleksitas lebih berat", "info"),
        kpi_card("Ljung-Box p-value", idn(info_model["lb_pvalue"], 4),
                 "✔ " + status_lb if lb_ok else "✖ " + status_lb, "up" if lb_ok else "down"),
    ]) + "</div>")

    pill = '<span class="pill ok">White Noise (Memenuhi Asumsi)</span>' if lb_ok \
        else '<span class="pill bad">Belum White Noise</span>'
    html(f"""
<div class="card">
<table class="spec">
<tr><th>Parameter</th><th>Nilai</th><th>Keterangan</th></tr>
<tr><td class="k">Orde model (p, d, q)</td><td class="v">{order_str}</td><td>p = komponen autoregresif, d = jumlah diferensiasi agar data stasioner, q = komponen moving average dari galat masa lalu.</td></tr>
<tr><td class="k">AIC</td><td class="v">{idn(info_model['aic'], 4)}</td><td>Akaike Information Criterion: menyeimbangkan kecocokan model dan kompleksitas. Dipakai memilih orde terbaik pada grid search (nilai terkecil terbaik).</td></tr>
<tr><td class="k">BIC</td><td class="v">{idn(info_model['bic'], 4)}</td><td>Bayesian Information Criterion: seperti AIC dengan penalti jumlah parameter yang lebih besar.</td></tr>
<tr><td class="k">Ljung-Box Stat (Lag 10)</td><td class="v">{idn(info_model['lb_stat'], 4)}</td><td>Statistik uji autokorelasi residual pada 10 lag pertama.</td></tr>
<tr><td class="k">Ljung-Box p-value</td><td class="v">{idn(info_model['lb_pvalue'], 4)}</td><td>Jika p &gt; 0,05 maka residual tidak berautokorelasi (white noise), artinya pola data sudah tertangkap model.</td></tr>
<tr><td class="k">Status Residual</td><td class="v">{pill}</td><td>Kesimpulan uji Ljung-Box pada taraf signifikansi 5%.</td></tr>
</table>
</div>
""")

    l, r = st.columns(2)
    with l:
        section("Metrik Evaluasi (Seluruh Data Uji)")
        st.dataframe(evaluasi.round(4), hide_index=True, **WIDE)
    with r:
        section("Top Kombinasi Grid Search (AIC Terkecil)")
        st.dataframe(gs_aic.round(4), hide_index=True, **WIDE)

    section("Arti Metrik Evaluasi")
    html("""
<div class="card">
<table class="spec">
<tr><th>Metrik</th><th>Arti</th><th>Cara membaca</th></tr>
<tr><td class="k">MAE</td><td>Rata-rata selisih absolut antara aktual dan prediksi (dalam poin indeks).</td><td>Makin kecil makin akurat; mudah ditafsirkan.</td></tr>
<tr><td class="k">RMSE</td><td>Akar dari rata-rata kuadrat galat; memberi bobot lebih besar pada galat besar.</td><td>Makin kecil makin baik; RMSE jauh di atas MAE menandakan ada galat ekstrem.</td></tr>
<tr><td class="k">MAPE</td><td>Rata-rata persentase galat terhadap nilai aktual.</td><td>&lt; 10% Sangat Akurat · 10–20% Akurat · 20–50% Cukup Akurat · &gt; 50% Tidak Akurat.</td></tr>
<tr><td class="k">Akurasi Arah</td><td>Persentase hari ketika arah prediksi (naik/turun) sama dengan arah aktual.</td><td>50% setara tebakan acak; makin tinggi makin baik.</td></tr>
</table>
</div>
""")

    section("5 Hari dengan Error Terbesar", "Pada rentang tanggal terpilih.")
    top = p.sort_values("Abs_Error", ascending=False).head(5)[
        ["Tanggal", "Aktual", "Prediksi", "Selisih", "Abs_Error", "APE (%)"]].copy()
    top["Tanggal"] = top["Tanggal"].dt.date
    st.dataframe(top.round(2), hide_index=True, **WIDE)

# ------------------------------------------------------------
# TAB 4: FORECAST KE DEPAN (MULTI-STEP ARIMA)
# ------------------------------------------------------------
with tab4:
    section(f"Forecast Harga Penutupan IHSG ke Depan ({order_str})",
            "Prediksi dihitung mulai hari bursa setelah tanggal terakhir pada dataset.")

    n_hari = st.slider("Jumlah hari bursa yang diprediksi ke depan", min_value=1, max_value=30, value=7)
    model_arima = load_model()

    fc_res = model_arima.get_forecast(steps=n_hari)
    fc_mean = np.asarray(fc_res.predicted_mean)
    fc_ci = np.asarray(fc_res.conf_int(alpha=0.05))

    last_date = full["Tanggal"].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_hari)

    df_fc = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi": fc_mean,
        "Batas Bawah (95%)": fc_ci[:, 0],
        "Batas Atas (95%)": fc_ci[:, 1],
    })

    last_close = float(full["Terakhir"].iloc[-1])
    fc_end = float(df_fc["Prediksi"].iloc[-1])
    fc_chg = (fc_end / last_close - 1) * 100
    html('<div class="kpi-grid-4">' + "".join([
        kpi_card("Penutupan Terakhir Dataset", idn(last_close), fmt_tgl(last_date.date()), "info"),
        kpi_card(f"Forecast Hari Ke-{n_hari}", idn(fc_end), fmt_tgl(df_fc["Tanggal"].iloc[-1].date()), "info"),
        kpi_card("Perubahan Proyeksi", f"{idn_signed(fc_chg)}%", "vs penutupan terakhir",
                 "up" if fc_chg >= 0 else "down"),
        kpi_card("Rentang CI 95% (Hari Terakhir)",
                 f"{idn(df_fc['Batas Bawah (95%)'].iloc[-1], 0)} – {idn(df_fc['Batas Atas (95%)'].iloc[-1], 0)}",
                 "Makin jauh horizon, makin lebar", "info"),
    ]) + "</div>")

    # Plot 30 hari terakhir + forecast
    hist = full.tail(30)
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=hist["Tanggal"], y=hist["Terakhir"], name="Aktual (30 Hari Terakhir)",
                               line=dict(color=NAVY, width=2.4), hovertemplate="%{y:,.2f}"))
    fig_f.add_trace(go.Scatter(
        x=pd.concat([df_fc["Tanggal"], df_fc["Tanggal"][::-1]]),
        y=pd.concat([df_fc["Batas Atas (95%)"], df_fc["Batas Bawah (95%)"][::-1]]),
        fill="toself", fillcolor="rgba(14, 165, 233, 0.16)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence Interval 95%", hoverinfo="skip",
    ))
    fig_f.add_trace(go.Scatter(x=df_fc["Tanggal"], y=df_fc["Prediksi"], name="Forecast ARIMA",
                               mode="lines+markers", line=dict(color=SKY, width=2.6, dash="dash"),
                               marker=dict(size=6, color=SKY), hovertemplate="%{y:,.2f}"))
    style_fig(fig_f, height=440, ytitle="IHSG")
    st.plotly_chart(fig_f, key="chart_forecast", **WIDE)

    df_fc_show = df_fc.copy()
    df_fc_show["Tanggal"] = df_fc_show["Tanggal"].dt.date
    st.dataframe(df_fc_show.round(2), hide_index=True, **WIDE)
    st.download_button(
        "⬇  Unduh hasil forecast (CSV)",
        data=df_fc_show.round(4).to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_ihsg_{n_hari}_hari.csv",
        mime="text/csv",
        key="dl_forecast",
    )

html('<div class="foot">Dashboard ini dibuat untuk keperluan akademik dan analisis statistik. '
     'Bukan merupakan saran investasi.</div>')
