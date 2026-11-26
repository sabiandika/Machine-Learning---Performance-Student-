import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import (
    WARNA_AKSEN,
    WARNA_AMAN,
    WARNA_RISIKO,
    ambil_pilihan_dataset,
    ambil_threshold,
    bagi_data,
    evaluasi_threshold,
    fmt,
    kotak,
    latih_semua_model,
    tata_letak_plotly,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Modelling & Evaluasi (R²)")
st.caption(f"Perbandingan 4 metode regresi dan evaluasi berbasis threshold pada **{pilihan}**")

_, X_test, _, y_test = bagi_data(pilihan)
tabel, fitted, prediksi = latih_semua_model(pilihan)

t1, t2, t3, t4 = st.tabs([
    "Metode & Konfigurasi",
    "Perbandingan R², RMSE, MAE",
    "Aktual vs Prediksi & Residual",
    f"Evaluasi Threshold (G3 ≥ {threshold:.1f})",
])

with t1:
    st.subheader("Metode Machine Learning yang Digunakan")
    info = [
        ("Random Forest (Tuned)", "Ensemble ratusan pohon keputusan (bagging) dengan rata-rata prediksi yang stabil dan tahan non-linearitas.", "n_estimators=600, max_features=0.7, min_samples_leaf=2", "Tidak perlu scaling"),
        ("Extra Trees", "Ensemble pohon yang memilih titik belah (split) secara lebih acak untuk menekan varians.", "n_estimators=400, max_features=0.8, min_samples_leaf=2", "Tidak perlu scaling"),
        ("Ridge Regression", "Regresi linear dengan regularisasi L2 untuk menangani multikolinearitas antara G1 dan G2.", "alpha=1.0", "Ya (StandardScaler)"),
        ("Decision Tree (Baseline)", "Pohon keputusan tunggal sebagai pembanding awal yang mudah diinterpretasikan.", "max_depth=5, min_samples_leaf=3", "Tidak perlu scaling"),
    ]
    cols = st.columns(2)
    for i, (nama, cara, konf, scale) in enumerate(info):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{nama}**")
                st.write(cara)
                st.caption(f"Konfigurasi: `{konf}` | Scaling: {scale}")

    kotak(
        "<b>Rumus & Makna R² (Koefisien Determinasi):</b><br>"
        "<code>R² = 1 - (SS_res / SS_tot)</code>. Nilai R² mendekati <b>1,0</b> berarti model mampu menjelaskan "
        "hampir seluruh variasi nilai akhir G3 siswa.",
        "info",
        "METRIK UTAMA: R²",
    )

with t2:
    st.subheader(f"Tabel Perbandingan Model ({len(y_test)} data uji)")
    st.dataframe(
        tabel.style.format({
            "CV R² Mean": "{:.4f}",
            "CV R² Std": "{:.4f}",
            "CV RMSE": "{:.3f}",
            "CV MAE": "{:.3f}",
            "Test R²": "{:.4f}",
            "Test RMSE": "{:.3f}",
            "Test MAE": "{:.3f}",
        })
        .highlight_max(subset=["CV R² Mean", "Test R²"], color="#b7e4c7")
        .highlight_min(subset=["CV RMSE", "CV MAE", "Test RMSE", "Test MAE"], color="#b7e4c7"),
        width="stretch",
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    fig_r2 = px.bar(
        tabel.sort_values("Test R²"),
        x="Test R²",
        y="Model",
        orientation="h",
        text=tabel.sort_values("Test R²")["Test R²"].map(lambda v: fmt(v, 4)),
        color="Test R²",
        color_continuous_scale="Tealgrn",
        title="Perbandingan Test R² Antar Model",
    )
    fig_r2.update_layout(coloraxis_showscale=False, xaxis=dict(range=[0, 1.05]))
    c1.plotly_chart(tata_letak_plotly(fig_r2, 340), width="stretch")

    fig_rmse = px.bar(
        tabel.sort_values("Test RMSE", ascending=False),
        x="Test RMSE",
        y="Model",
        orientation="h",
        text=tabel.sort_values("Test RMSE", ascending=False)["Test RMSE"].map(lambda v: fmt(v, 3)),
        color="Test RMSE",
        color_continuous_scale="Reds",
        title="Perbandingan Test RMSE (Lebih Kecil Lebih Baik)",
    )
    fig_rmse.update_layout(coloraxis_showscale=False)
    c2.plotly_chart(tata_letak_plotly(fig_rmse, 340), width="stretch")

with t3:
    model_pilih = st.selectbox("Pilih model untuk inspeksi residual", tabel["Model"].tolist(), index=0)
    y_pred = prediksi[model_pilih]
    residual = y_test.to_numpy() - y_pred

    c1, c2 = st.columns(2)
    fig_act = go.Figure()
    fig_act.add_scatter(x=y_test, y=y_pred, mode="markers", marker=dict(color=WARNA_AKSEN, opacity=0.65), name="Sampel Uji")
    fig_act.add_scatter(x=[0, 20], y=[0, 20], mode="lines", line=dict(color=WARNA_RISIKO, dash="dash"), name="Ideal (y = x)")
    fig_act.update_layout(title=f"G3 Aktual vs Prediksi ({model_pilih})", xaxis_title="G3 Aktual", yaxis_title="G3 Prediksi")
    c1.plotly_chart(tata_letak_plotly(fig_act, 380), width="stretch")

    fig_res = go.Figure()
    fig_res.add_scatter(x=y_pred, y=residual, mode="markers", marker=dict(color=WARNA_AMAN, opacity=0.65), name="Residual")
    fig_res.add_hline(y=0, line_dash="dash", line_color=WARNA_RISIKO)
    fig_res.update_layout(title=f"Residual Plot ({model_pilih})", xaxis_title="G3 Prediksi", yaxis_title="Residual (Aktual - Prediksi)")
    c2.plotly_chart(tata_letak_plotly(fig_res, 380), width="stretch")

    err_df = X_test[["G1", "G2", "absences", "failures"]].copy()
    if "subject" in X_test.columns:
        err_df.insert(0, "subject", X_test["subject"])
    err_df["G3 Aktual"] = y_test.to_numpy()
    err_df["G3 Prediksi"] = np.round(y_pred, 2)
    err_df["Absolute Error"] = np.round(np.abs(residual), 2)
    st.markdown("**10 error prediksi terbesar pada data uji**")
    st.dataframe(err_df.sort_values("Absolute Error", ascending=False).head(10), width="stretch", hide_index=True)

with t4:
    st.subheader(f"Dampak Threshold Kelulusan (G3 ≥ {threshold:.1f}) pada Data Uji")
    st.write(
        "Geser slider **Threshold kelulusan G3** di sidebar untuk melihat perubahan Recall siswa berisiko "
        "(`G3 < threshold`), Precision, dan jumlah False Positive / False Negative secara langsung."
    )

    th_rows = []
    for nama in tabel["Model"]:
        m = evaluasi_threshold(y_test, prediksi[nama], threshold)
        th_rows.append({"Model": nama, **m})
    df_th = pd.DataFrame(th_rows)
    st.dataframe(
        df_th.style.format({
            "Accuracy": "{:.4f}",
            "Recall Risiko": "{:.4f}",
            "Precision Risiko": "{:.4f}",
            "F1 Risiko": "{:.4f}",
        }).highlight_max(subset=["Recall Risiko", "F1 Risiko"], color="#b7e4c7"),
        width="stretch",
        hide_index=True,
    )

    model_cm = st.selectbox("Tampilkan Confusion Matrix untuk model:", tabel["Model"].tolist(), index=0, key="cm_model")
    m_sel = evaluasi_threshold(y_test, prediksi[model_cm], threshold)
    cm_mat = np.array([[m_sel["TN"], m_sel["FP"]], [m_sel["FN"], m_sel["TP"]]])
    cA, cB = st.columns([1.1, 1])
    fig_cm = px.imshow(
        cm_mat,
        text_auto=True,
        x=[f"Prediksi Lulus (≥{threshold:.1f})", f"Prediksi Berisiko (<{threshold:.1f})"],
        y=[f"Aktual Lulus (≥{threshold:.1f})", f"Aktual Berisiko (<{threshold:.1f})"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {model_cm}",
    )
    fig_cm.update_layout(coloraxis_showscale=False)
    cA.plotly_chart(tata_letak_plotly(fig_cm, 340), width="stretch")

    with cB:
        st.metric("Recall Siswa Berisiko", fmt(m_sel["Recall Risiko"], 4))
        st.metric("False Negative (Berisiko tapi diprediksi Lulus)", m_sel["FN"])
        st.metric("False Positive (Lulus tapi ditandai Berisiko)", m_sel["FP"])
        st.caption("Semakin tinggi threshold, semakin banyak siswa tertandai berisiko (Recall naik, tetapi FP juga berpotensi naik).")
