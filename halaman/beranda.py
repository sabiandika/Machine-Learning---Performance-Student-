import streamlit as st
from utils import (
    ambil_pilihan_dataset,
    ambil_threshold,
    bagi_data,
    evaluasi_threshold,
    fmt,
    hero,
    kotak,
    latih_semua_model,
    ringkas_kualitas_data,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

hero(
    "Prediksi Performa Siswa (Student Performance)",
    "Eksplorasi Data (EDA), Perbandingan Fitur, dan Pemodelan Regresi R² untuk memprediksi nilai akhir G3 (0–20)",
)

q = ringkas_kualitas_data(pilihan, threshold)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah sampel siswa", q["baris"])
k2.metric("Jumlah fitur prediktor", q["fitur"])
k3.metric("Missing value", q["missing"])
k4.metric(f"Siswa lulus (G3 ≥ {threshold:.1f})", f"{q['lulus']} ({fmt(q['rasio_lulus'], 1)}%)")

st.subheader("Tentang proyek & metode yang digunakan")
st.write(
    f"Aplikasi ini menganalisis dataset **Student Performance** pada pilihan **{pilihan}**. "
    "Tujuan utamanya adalah memprediksi nilai akhir siswa **G3 (rentang 0–20)** menggunakan pendekatan "
    "**Machine Learning Regresi** dengan metode utama **Random Forest Regressor (Tuned)**, "
    "yang dibandingkan terhadap **Extra Trees**, **Ridge Regression**, dan **Decision Tree (Baseline)** "
    "dalam satu `Pipeline` preprocessing (`SimpleImputer` + `OneHotEncoder`)."
)
kotak(
    "<b>Skenario & Interpretasi Threshold:</b> Model memprediksi nilai akhir <b>G3</b> setelah nilai periode "
    "<b>G1</b> dan <b>G2</b> tersedia. Dengan <b>Threshold Kelulusan G3</b> di sidebar (saat ini "
    f"<b>{threshold:.1f}</b>), prediksi regresi juga dipetakan menjadi status <b>Lulus (Aman)</b> vs "
    "<b>Berisiko Tidak Lulus</b> agar sekolah dapat mendeteksi siswa yang membutuhkan pendampingan lebih awal.",
    "info",
    "METODE & THRESHOLD",
)

st.subheader("Alur aplikasi")
alur = [
    ("1. Dataset", "Cerita dataset Matematika & Portugis, struktur 33 variabel, dan data mentah"),
    ("2. EDA", "Kualitas data, distribusi, korelasi ke G3, dan 10 masalah kritis EDA"),
    ("3. Preprocessing", "Split 80/20 stratified, imputasi, OneHotEncoder tanpa kebocoran"),
    ("4. Perbandingan Fitur", "Tombol uji Semua Variabel vs Fitur Penting Saja & Tanpa G1/G2"),
    ("5. Modelling (R²)", "Evaluasi R², RMSE, MAE, residual plot, dan metrik threshold"),
    ("6. Prediksi Siswa", "Prediksi G3 satu siswa + rentang 80% + status vs threshold"),
    ("7. Prediksi Massal", "Unggah file CSV siswa untuk prediksi dan unduh hasil"),
    ("8. Kesimpulan", "Temuan utama, keterbatasan, dan rekomendasi pengembangan"),
]
cols = st.columns(4)
for i, (j, d) in enumerate(alur):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{j}**")
            st.caption(d)

st.subheader(f"Hasil utama model pada {pilihan}")
tabel, _, prediksi = latih_semua_model(pilihan)
_, X_test, _, y_test = bagi_data(pilihan)
terbaik = tabel.iloc[0]
rf_row = tabel[tabel["Model"] == "Random Forest (Tuned)"].iloc[0]
dt_row = tabel[tabel["Model"] == "Decision Tree (Baseline)"].iloc[0]
th_eval = evaluasi_threshold(y_test, prediksi["Random Forest (Tuned)"], threshold)

a, b, c, d = st.columns(4)
a.metric("Test R² (Random Forest)", fmt(rf_row["Test R²"], 4), f"{rf_row['Test R²'] - dt_row['Test R²']:+.4f} vs Decision Tree")
b.metric("Test RMSE", fmt(rf_row["Test RMSE"], 3), f"{rf_row['Test RMSE'] - dt_row['Test RMSE']:+.3f} vs Decision Tree", delta_color="inverse")
c.metric("Test MAE", fmt(rf_row["Test MAE"], 3))
d.metric(f"Recall Risiko (G3 < {threshold:.1f})", fmt(th_eval["Recall Risiko"], 3), f"{th_eval['FP']} FP / {th_eval['FN']} FN")

st.caption(
    f"Model terbaik berdasarkan Test R² pada dataset aktif: **{terbaik['Model']}** (R² = {fmt(terbaik['Test R²'], 4)}). "
    "Ubah pilihan dataset atau slider threshold di sidebar untuk melihat dampaknya secara langsung."
)
