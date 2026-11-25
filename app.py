import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path("artifacts")
st.set_page_config(page_title="Dashboard Prediksi IHSG (ARIMA)", page_icon="📈", layout="wide")

HIJAU, MERAH, BIRU, ORANGE = "#2e9e5b", "#d64545", "#1f77b4", "#ff7f0e"


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


def kategori_mape(v):
    if v < 10:
        return "Sangat Akurat"
    if v <= 20:
        return "Akurat"
    if v <= 50:
        return "Cukup Akurat"
    return "Tidak Akurat"


pred, full, evaluasi, gs_aic, info_model, split = load_data()
order_str = f"ARIMA({info_model['p']}, {info_model['d']}, {info_model['q']})"

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("⚙️ Filter")
tmin, tmax = pred["Tanggal"].min().date(), pred["Tanggal"].max().date()
rentang = st.sidebar.date_input("Rentang tanggal (data uji)", (tmin, tmax), min_value=tmin, max_value=tmax)
if isinstance(rentang, tuple) and len(rentang) == 2:
    d1, d2 = rentang
else:
    d1, d2 = tmin, tmax
p = pred[(pred["Tanggal"].dt.date >= d1) & (pred["Tanggal"].dt.date <= d2)].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Model:** {order_str}\n\n"
    f"**Data latih:** {split['train_start']} s/d {split['train_end']} ({split['n_train']} hari)\n\n"
    f"**Data uji:** {split['test_start']} s/d {split['test_end']} ({split['n_test']} hari)"
)

# ------------------------------------------------------------
# HEADER & KPI
# ------------------------------------------------------------
st.title("📈 Dashboard Prediksi Harian IHSG")
st.caption(f"Prediksi harga penutupan IHSG di masa pemerintahan Prabowo–Gibran menggunakan {order_str}")

if p.empty:
    st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    st.stop()

mae = p["Selisih"].abs().mean()
rmse = np.sqrt((p["Selisih"] ** 2).mean())
mape = (p["Selisih"].abs() / p["Aktual"]).mean() * 100
akurasi_arah = p["Arah_Tepat"].mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("MAPE", f"{mape:.2f}%", kategori_mape(mape), delta_color="off")
c4.metric("Akurasi Arah", f"{akurasi_arah:.1f}%")
c5.metric("Penutupan Terakhir", f"{p['Aktual'].iloc[-1]:,.2f}", f"{p['Aktual_Chg%'].iloc[-1]:+.2f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Aktual vs Prediksi", "🔼🔽 Naik / Turun Harian", "🧪 Evaluasi & Model", "🔮 Forecast Ke Depan"]
)

# ------------------------------------------------------------
# TAB 1: AKTUAL VS PREDIKSI
# ------------------------------------------------------------
with tab1:
    st.subheader(f"Harga Penutupan IHSG: Aktual vs Prediksi {order_str}")
    tampil_latih = st.checkbox("Tampilkan data latih (riwayat sebelum periode uji)", value=False)

    fig = go.Figure()
    if tampil_latih:
        latih = full[full["Tanggal"] < pd.to_datetime(split["test_start"])]
        fig.add_trace(go.Scatter(x=latih["Tanggal"], y=latih["Terakhir"], name="Data latih",
                                 line=dict(color="gray", width=1.5)))
    fig.add_trace(go.Scatter(x=p["Tanggal"], y=p["Aktual"], name="Aktual",
                             line=dict(color=BIRU, width=2.5)))
    fig.add_trace(go.Scatter(x=p["Tanggal"], y=p["Prediksi"], name=f"Prediksi {order_str}",
                             line=dict(color=ORANGE, width=2.5, dash="dash")))
    fig.update_layout(height=450, hovermode="x unified", yaxis_title="IHSG",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selisih Prediksi (Aktual − Prediksi)")
    warna = np.where(p["Selisih"] >= 0, HIJAU, MERAH)
    fig_e = go.Figure(go.Bar(x=p["Tanggal"], y=p["Selisih"], marker_color=warna))
    fig_e.update_layout(height=280, yaxis_title="Poin", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_e, use_container_width=True)
    st.caption("Hijau: model memprediksi lebih rendah dari aktual (under-predict). "
               "Merah: model memprediksi lebih tinggi dari aktual (over-predict).")

# ------------------------------------------------------------
# TAB 2: NAIK / TURUN HARIAN
# ------------------------------------------------------------
with tab2:
    st.subheader("Perubahan Harian Prediksi vs Aktual (%)")
    st.caption("Perubahan dihitung terhadap harga penutupan hari bursa sebelumnya.")

    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(x=p["Tanggal"], y=p["Prediksi_Chg%"], name="Prediksi",
                           marker_color=np.where(p["Prediksi_Chg%"] >= 0, HIJAU, MERAH)))
    fig_c.add_trace(go.Scatter(x=p["Tanggal"], y=p["Aktual_Chg%"], name="Aktual",
                               mode="lines+markers", line=dict(color="black", width=1.5),
                               marker=dict(size=4)))
    fig_c.update_layout(height=420, yaxis_title="Perubahan (%)", hovermode="x unified",
                        legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_c, use_container_width=True)

    a, b = st.columns([1, 2])
    with a:
        st.markdown("**Ringkasan arah prediksi**")
        naik_p = (p["Arah_Prediksi"] == "Naik").sum()
        turun_p = (p["Arah_Prediksi"] == "Turun").sum()
        naik_a = (p["Arah_Aktual"] == "Naik").sum()
        turun_a = (p["Arah_Aktual"] == "Turun").sum()
        st.dataframe(pd.DataFrame({"Prediksi": [naik_p, turun_p], "Aktual": [naik_a, turun_a]},
                                  index=["Naik", "Turun"]), use_container_width=True)
        st.metric("Arah tepat", f"{p['Arah_Tepat'].sum()} dari {len(p)} hari")
    with b:
        st.markdown("**Confusion matrix arah (baris: aktual, kolom: prediksi)**")
        cm = pd.crosstab(p["Arah_Aktual"], p["Arah_Prediksi"]).reindex(
            index=["Naik", "Turun"], columns=["Naik", "Turun"], fill_value=0)
        st.dataframe(cm, use_container_width=True)

    st.markdown("**Detail harian**")
    detail = p[["Tanggal", "Prev", "Aktual", "Prediksi", "Aktual_Chg%", "Prediksi_Chg%",
                "Arah_Aktual", "Arah_Prediksi", "Arah_Tepat"]].copy()
    detail["Tanggal"] = detail["Tanggal"].dt.date
    st.dataframe(detail.round(2).rename(columns={"Prev": "Penutupan Sebelumnya"}),
                 use_container_width=True, height=320)

# ------------------------------------------------------------
# TAB 3: EVALUASI & MODEL ARIMA
# ------------------------------------------------------------
with tab3:
    l, r = st.columns(2)
    with l:
        st.subheader("Metrik Evaluasi (Seluruh Data Uji)")
        st.dataframe(evaluasi.round(4), use_container_width=True, hide_index=True)

        st.subheader("Spesifikasi & Diagnostik Model")
        status_lb = "White Noise (Memenuhi Asumsi)" if info_model["lb_pvalue"] > 0.05 else "Belum White Noise"
        st.json({
            "Model": order_str,
            "AIC": round(info_model["aic"], 4),
            "BIC": round(info_model["bic"], 4),
            "Ljung-Box Stat (Lag 10)": round(info_model["lb_stat"], 4),
            "Ljung-Box p-value": round(info_model["lb_pvalue"], 4),
            "Status Residual": status_lb,
        })
    with r:
        st.subheader("Top Kombinasi Grid Search (AIC Terkecil)")
        st.dataframe(gs_aic.round(4), use_container_width=True, hide_index=True)

    st.subheader("5 Hari dengan Error Terbesar")
    top = p.sort_values("Abs_Error", ascending=False).head(5)[
        ["Tanggal", "Aktual", "Prediksi", "Selisih", "Abs_Error", "APE (%)"]].copy()
    top["Tanggal"] = top["Tanggal"].dt.date
    st.dataframe(top.round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 4: FORECAST KE DEPAN (MULTI-STEP ARIMA)
# ------------------------------------------------------------
with tab4:
    st.subheader(f"Forecast Harga Penutupan IHSG ke Depan ({order_str})")
    st.caption("Prediksi dihitung mulai hari bursa setelah tanggal terakhir pada dataset.")

    n_hari = st.slider("Jumlah hari bursa yang diprediksi ke depan", min_value=1, max_value=30, value=7)
    model_arima = load_model()

    fc_res = model_arima.get_forecast(steps=n_hari)
    fc_mean = fc_res.predicted_mean
    fc_ci = fc_res.conf_int(alpha=0.05)

    last_date = full["Tanggal"].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_hari)

    df_fc = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi": fc_mean,
        "Batas Bawah (95%)": fc_ci[:, 0],
        "Batas Atas (95%)": fc_ci[:, 1],
    })

    # Plot 30 hari terakhir + forecast
    hist = full.tail(30)
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=hist["Tanggal"], y=hist["Terakhir"], name="Aktual (30 Hari Terakhir)",
                               line=dict(color=BIRU, width=2.2)))
    fig_f.add_trace(go.Scatter(x=df_fc["Tanggal"], y=df_fc["Prediksi"], name="Forecast ARIMA",
                               line=dict(color=ORANGE, width=2.5, dash="dash"), mode="lines+markers"))
    fig_f.add_trace(go.Scatter(
        x=pd.concat([df_fc["Tanggal"], df_fc["Tanggal"][::-1]]),
        y=pd.concat([df_fc["Batas Atas (95%)"], df_fc["Batas Bawah (95%)"][::-1]]),
        fill="toself", fillcolor="rgba(255, 127, 14, 0.18)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence Interval 95%"
    ))
    fig_f.update_layout(height=420, hovermode="x unified", yaxis_title="IHSG",
                        legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_f, use_container_width=True)

    df_fc_show = df_fc.copy()
    df_fc_show["Tanggal"] = df_fc_show["Tanggal"].dt.date
    st.dataframe(df_fc_show.round(2), use_container_width=True, hide_index=True)
