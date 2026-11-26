import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (
    WARNA_AMAN,
    WARNA_RISIKO,
    ambil_pilihan_dataset,
    ambil_threshold,
    bagi_data,
    buat_preprocessor,
    fmt,
    kotak,
    siapkan_xy,
    tata_letak_plotly,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Preprocessing Data")
st.caption(f"Pembersihan, pembagian data 80/20 stratified, dan `ColumnTransformer` pada **{pilihan}**")

X, y = siapkan_xy(pilihan)
X_train, X_test, y_train, y_test = bagi_data(pilihan)

a, b, c, d = st.columns(4)
a.metric("Data latih (80%)", len(X_train))
b.metric("Data uji (20%)", len(X_test))
c.metric("Fitur sebelum encoding", X_train.shape[1])

prep = buat_preprocessor(X_train, scale_numeric=False)
X_tr_enc = prep.fit_transform(X_train)
d.metric("Fitur setelah OneHotEncoder", X_tr_enc.shape[1])

st.subheader("Alur Pipeline Preprocessing")
langkah = [
    ("1. Pemisahan Target", "Kolom `G3` (0–20) dipisahkan sebagai target regresi; fitur lainnya menjadi prediktor `X`."),
    ("2. Stratified Split 80/20", "Pembagian train/test distratifikasi berdasarkan kelompok nilai (`low`: 0–9, `medium`: 10–14, `high`: 15–20) agar proporsi seimbang."),
    ("3. Imputasi Aman", "Fitur numerik diimputasi dengan median; fitur kategorikal dengan modus (`most_frequent`) hanya di-fit pada train set."),
    ("4. OneHotEncoder", "`OneHotEncoder(handle_unknown='ignore')` mengubah fitur kategorikal menjadi biner tanpa error jika ada kategori baru."),
]
cols = st.columns(4)
for col, (judul, isi) in zip(cols, langkah):
    with col:
        with st.container(border=True):
            st.markdown(f"**{judul}**")
            st.caption(isi)

kotak(
    "<b>Mengapa outlier `absences` tidak dihapus?</b><br>"
    "Siswa dengan ketidakhadiran tinggi adalah kasus nyata di lapangan, bukan kesalahan ketik. "
    "Model berbasis pohon (Random Forest & Extra Trees) secara alami tahan terhadap outlier tersebut. "
    "Selain itu, hasil prediksi model di-clip ke rentang valid <b>0–20</b>.",
    "info",
    "KEBIJAKAN OUTLIER & DOMAIN",
)

st.subheader("Verifikasi proporsi Train vs Test berdasarkan Threshold")
df_split = pd.DataFrame([
    {
        "Split": "Train (80%)",
        "Jumlah": len(y_train),
        "Rata-rata G3": float(y_train.mean()),
        f"% Lulus (≥ {threshold:.1f})": float((y_train >= threshold).mean() * 100),
    },
    {
        "Split": "Test (20%)",
        "Jumlah": len(y_test),
        "Rata-rata G3": float(y_test.mean()),
        f"% Lulus (≥ {threshold:.1f})": float((y_test >= threshold).mean() * 100),
    },
])
st.dataframe(df_split.round(2), width="stretch", hide_index=True)

fig_dist = px.histogram(
    pd.DataFrame({
        "G3": pd.concat([y_train, y_test], ignore_index=True),
        "Split": ["Train (80%)"] * len(y_train) + ["Test (20%)"] * len(y_test),
    }),
    x="G3",
    color="Split",
    barmode="overlay",
    histnorm="probability density",
    opacity=0.65,
    color_discrete_map={"Train (80%)": WARNA_AMAN, "Test (20%)": WARNA_RISIKO},
    title="Perbandingan Kepadatan Distribusi G3 pada Train vs Test",
)
st.plotly_chart(tata_letak_plotly(fig_dist, 340), width="stretch")
st.caption(f"Selisih rata-rata G3 antara Train dan Test hanya {fmt(abs(y_train.mean() - y_test.mean()), 3)} poin.")
