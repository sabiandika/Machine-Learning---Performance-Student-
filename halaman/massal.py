import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    ambil_pilihan_dataset,
    ambil_threshold,
    latih_semua_model,
    muat_artefak_rf,
    prediksi_dengan_interval,
    siapkan_xy,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Prediksi Massal (CSV)")
st.caption(f"Unggah file CSV siswa untuk memprediksi G3 sekaligus pada **{pilihan}**")

X, _ = siapkan_xy(pilihan)
_, fitted, _ = latih_semua_model(pilihan)
artefak = muat_artefak_rf()

contoh_csv = X.head(10).to_csv(index=False).encode("utf-8")
st.download_button(
    "Unduh contoh template CSV (10 baris)",
    data=contoh_csv,
    file_name="contoh_data_siswa.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Unggah file CSV (separator koma `,` atau titik koma `;`)", type=["csv"])
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

    missing_cols = [c for c in X.columns if c not in df_in.columns]
    if missing_cols:
        st.error(f"Kolom wajib belum lengkap: {', '.join(missing_cols)}")
        st.stop()

    pipe = artefak["pipeline"] if (pilihan.startswith("Gabungan") and artefak is not None) else fitted["Random Forest (Tuned)"]
    X_in = df_in[X.columns].copy()
    pred, lo80, hi80 = prediksi_dengan_interval(pipe, X_in)

    hasil = df_in.copy()
    hasil["Prediksi_G3"] = np.round(pred, 2)
    hasil["Batas_Bawah_80%"] = np.round(lo80, 2)
    hasil["Batas_Atas_80%"] = np.round(hi80, 2)
    hasil["Status_Kelulusan"] = np.where(pred >= threshold, f"Lulus (≥ {threshold:.1f})", f"Berisiko (< {threshold:.1f})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total siswa diprediksi", len(hasil))
    c2.metric(f"Diprediksi Lulus (≥ {threshold:.1f})", int((pred >= threshold).sum()))
    c3.metric(f"Diprediksi Berisiko (< {threshold:.1f})", int((pred < threshold).sum()))

    st.dataframe(hasil.head(50), width="stretch", hide_index=True)
    st.download_button(
        "Unduh hasil prediksi (CSV)",
        data=hasil.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_siswa.csv",
        mime="text/csv",
        type="primary",
    )
