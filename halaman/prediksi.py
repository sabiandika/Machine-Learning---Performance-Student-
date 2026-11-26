import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    WARNA_AMAN,
    WARNA_RISIKO,
    ambil_pilihan_dataset,
    ambil_threshold,
    fmt,
    kotak,
    latih_semua_model,
    muat_artefak_rf,
    prediksi_dengan_interval,
    siapkan_xy,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Prediksi Satu Siswa")
st.caption(f"Simulasi prediksi nilai akhir **G3 (0–20)** dan status terhadap threshold (G3 ≥ {threshold:.1f}) pada **{pilihan}**")

X, y = siapkan_xy(pilihan)
tabel, fitted, _ = latih_semua_model(pilihan)
artefak = muat_artefak_rf()

opsi_model = list(fitted.keys())
if pilihan.startswith("Gabungan") and artefak is not None:
    opsi_model = ["Random Forest Produksi (.joblib)"] + opsi_model

c1, c2 = st.columns([1.4, 1])
nama_model = c1.selectbox("Pilih model prediksi", opsi_model)
c2.metric("Threshold kelulusan aktif", f"G3 ≥ {threshold:.1f}", help="Atur di sidebar kiri")

# Siapkan contoh default
def ambil_contoh(mask):
    sub = X[mask]
    row = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            row[col] = int(round(sub[col].median()))
        else:
            row[col] = str(sub[col].mode().iloc[0])
    return row


def isi_form(contoh):
    for col, val in contoh.items():
        st.session_state[f"pred_{col}"] = val


if f"pred_G1" not in st.session_state or st.session_state.get("pred_ds") != pilihan:
    st.session_state["pred_ds"] = pilihan
    isi_form(ambil_contoh(pd.Series(True, index=X.index)))

b1, b2, b3, _ = st.columns([1, 1, 1, 1.5])
if b1.button("Contoh Siswa Lulus", width="stretch"):
    isi_form(ambil_contoh(y >= threshold))
if b2.button("Contoh Siswa Berisiko", width="stretch"):
    isi_form(ambil_contoh(y < threshold))
if b3.button("Nilai Median/Modus", width="stretch"):
    isi_form(ambil_contoh(pd.Series(True, index=X.index)))

st.subheader("Masukkan data siswa")
tab1, tab2, tab3 = st.tabs(["Akademik & Absensi (Fitur Utama)", "Sekolah & Keluarga", "Aktivitas & Gaya Hidup"])

grup_akademik = [c for c in ["subject", "G1", "G2", "failures", "absences", "studytime", "schoolsup", "famsup", "paid", "higher"] if c in X.columns]
grup_keluarga = [c for c in ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "nursery"] if c in X.columns]
grup_gaya = [c for c in ["traveltime", "activities", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health"] if c in X.columns]


def render_inputs(cols_list):
    grid = st.columns(4)
    for i, col in enumerate(cols_list):
        with grid[i % 4]:
            if pd.api.types.is_numeric_dtype(X[col]):
                lo, hi = int(X[col].min()), int(X[col].max())
                st.number_input(col, min_value=lo, max_value=max(hi, 93 if col == "absences" else hi), step=1, key=f"pred_{col}")
            else:
                opts = sorted(X[col].astype(str).unique().tolist())
                st.selectbox(col, opts, key=f"pred_{col}")


with tab1:
    render_inputs(grup_akademik)
with tab2:
    render_inputs(grup_keluarga)
with tab3:
    render_inputs(grup_gaya)

input_row = pd.DataFrame([{col: st.session_state[f"pred_{col}"] for col in X.columns}])

st.subheader("Hasil Prediksi")
if st.button("Prediksi Nilai G3", type="primary"):
    if nama_model == "Random Forest Produksi (.joblib)" and artefak is not None:
        pipe = artefak["pipeline"]
        r2_info = artefak["test_metrics"]["R2"]
    else:
        pipe = fitted[nama_model]
        r2_info = float(tabel.loc[tabel["Model"] == nama_model, "Test R²"].iloc[0])

    pred, lo80, hi80 = prediksi_dengan_interval(pipe, input_row)
    nilai_g3 = float(pred[0])
    lulus = nilai_g3 >= threshold

    a, b = st.columns([1.1, 1.3])
    with a:
        if lulus:
            kotak(
                f"<h2 style='margin:0;color:{WARNA_AMAN}'>LULUS / AMAN</h2>"
                f"Prediksi G3: <b>{fmt(nilai_g3, 2)}</b> / 20 (Threshold ≥ {threshold:.1f})",
                "sesudah",
                "STATUS PREDIKSI",
            )
        else:
            kotak(
                f"<h2 style='margin:0;color:{WARNA_RISIKO}'>BERISIKO TIDAK LULUS</h2>"
                f"Prediksi G3: <b>{fmt(nilai_g3, 2)}</b> / 20 (di bawah Threshold {threshold:.1f})",
                "sebelum",
                "STATUS PREDIKSI",
            )
    with b:
        st.metric("Rentang ketidakpastian 80%", f"{fmt(float(lo80[0]), 2)} – {fmt(float(hi80[0]), 2)}")
        st.progress(min(max(nilai_g3 / 20.0, 0.0), 1.0), text=f"Skor prediksi {fmt(nilai_g3, 2)} / 20 | R² Model = {fmt(r2_info, 4)}")

    st.info(
        "Hasil prediksi merupakan estimasi statistik alat bantu keputusan akademik, bukan keputusan mutlak kelulusan.",
        icon=":material/info:",
    )
