import streamlit as st
from utils import (
    ambil_pilihan_dataset,
    ambil_threshold,
    bandingkan_skenario_fitur,
    fmt,
    kotak,
    latih_semua_model,
    ringkas_kualitas_data,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Kesimpulan & Saran")
st.caption(f"Ringkasan hasil pemodelan pada **{pilihan}** (Threshold aktif: G3 ≥ {threshold:.1f})")

q = ringkas_kualitas_data(pilihan, threshold)
tabel, _, _ = latih_semua_model(pilihan)
df_banding, _, _ = bandingkan_skenario_fitur(pilihan, top_k=8)
terbaik = tabel.iloc[0]
rf_row = tabel[tabel["Model"] == "Random Forest (Tuned)"].iloc[0]
row_semua = df_banding.iloc[0]
row_penting = df_banding.iloc[1]
row_early = df_banding.iloc[2]

st.header("Kesimpulan Utama")
c1, c2 = st.columns(2)
with c1:
    kotak(
        f"<b>1. Performa Regresi & R² Tinggi</b><br>"
        f"Model terbaik pada <b>{pilihan}</b> adalah <b>{terbaik['Model']}</b> dengan Test R² = "
        f"<b>{fmt(terbaik['Test R²'], 4)}</b>, RMSE = <b>{fmt(terbaik['Test RMSE'], 3)}</b>, dan MAE = <b>{fmt(terbaik['Test MAE'], 3)}</b>.",
        "sesudah",
    )
    kotak(
        f"<b>3. Efektivitas Seleksi Fitur</b><br>"
        f"Memakai semua variabel menghasilkan Test R² <b>{fmt(row_semua['Test R²'], 4)}</b>, sedangkan cukup 8 fitur terpenting "
        f"sudah mencapai R² <b>{fmt(row_penting['Test R²'], 4)}</b>. Namun jika G1 & G2 dibuang, R² turun ke <b>{fmt(row_early['Test R²'], 4)}</b>.",
        "sesudah",
    )
with c2:
    kotak(
        "<b>2. 10 Masalah Kritis EDA Teratasi</b><br>"
        "Kebocoran data dicegah lewat <code>Pipeline</code>, multikolinearitas G1–G2 ditangani oleh ensemble tree & Ridge, "
        "serta outlier valid pada <code>absences</code> dipertahankan.",
        "sesudah",
    )
    kotak(
        f"<b>4. Fleksibilitas Threshold Kelulusan</b><br>"
        f"Threshold G3 ({threshold:.1f}) di sidebar memungkinkan pihak sekolah mengatur keseimbangan antara "
        "mendeteksi sebanyak mungkin siswa berisiko (Recall tinggi) dan meminimalkan alarm palsu (False Positive).",
        "info",
    )

st.header("Keterbatasan & Saran Lanjutan")
a, b = st.columns(2)
with a:
    st.subheader("Keterbatasan")
    st.markdown(
        f"- Jumlah sampel pada {pilihan} adalah {q['baris']} baris, sehingga varians antar-split masih mungkin terjadi.\n"
        "- Model utama bergantung kuat pada nilai periode `G1` dan `G2`.\n"
        "- Faktor eksternal mendadak menjelang ujian akhir tidak terekam di dataset."
    )
with b:
    st.subheader("Saran")
    st.markdown(
        "- Jika ingin melakukan *early warning* di awal semester (sebelum ada G1 & G2), latih model terpisah tanpa G1/G2.\n"
        "- Sesuaikan threshold G3 sesuai kapasitas program bimbingan sekolah.\n"
        "- Lakukan kalibrasi ulang secara berkala setiap tahun ajaran baru."
    )
