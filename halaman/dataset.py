import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (
    DESKRIPSI_FITUR,
    WARNA_AMAN,
    WARNA_RISIKO,
    ambil_pilihan_dataset,
    ambil_threshold,
    fmt,
    kotak,
    muat_data,
    ringkas_kualitas_data,
    tata_letak_plotly,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Tentang Dataset")
st.caption(f"UCI Student Performance Dataset — Pilihan aktif: **{pilihan}**")

df = muat_data(pilihan)
q = ringkas_kualitas_data(pilihan, threshold)

t1, t2, t3, t4 = st.tabs(["Latar belakang & cerita data", "Struktur fitur", "Data mentah", "Distribusi nilai G3"])

with t1:
    c1, c2, c3 = st.columns(3)
    with c1:
        kotak(
            "Dataset berasal dari pencatatan dua sekolah menengah di Portugal (Gabriel Pereira & Mousinho da Silveira) "
            "yang dikumpulkan oleh Cortez & Silva (2008). Dataset terdiri dari dua berkas: nilai <b>Matematika</b> "
            "(395 siswa) dan nilai <b>Bahasa Portugis</b> (649 siswa), atau 1.044 baris jika digabungkan.",
            "info",
            "CERITA DATASET",
        )
    with c2:
        kotak(
            "Setiap siswa direkam dalam 32 atribut (demografi keluarga, kebiasaan belajar, aktivitas sosial, "
            "absensi, serta nilai periode <b>G1</b> dan <b>G2</b>) untuk memprediksi nilai akhir <b>G3 (0–20)</b>. "
            "Pada data gabungan ditambahkan kolom <code>subject</code> agar konteks mata pelajaran tetap terjaga.",
            "info",
            "MASALAH",
        )
    with c3:
        kotak(
            "Membangun model regresi yang akurat (<b>R² tinggi, RMSE & MAE rendah</b>) untuk memprediksi G3, "
            "sekaligus menyediakan <b>threshold kelulusan</b> agar siswa berisiko tidak lulus dapat teridentifikasi dini.",
            "info",
            "TUJUAN",
        )

    st.subheader("Apakah semua fitur dipakai oleh model?")
    st.write(
        "Secara default, seluruh fitur dalam dataset (**32 fitur** pada Matematika/Portugis tunggal, atau **33 fitur** "
        "dengan `subject` pada data gabungan) dimasukkan ke dalam pipeline preprocessing (`ColumnTransformer`). "
        "Namun, **tidak semua fitur memiliki kontribusi yang sama besar**—nilai `G2` dan `G1` menyumbang porsi "
        "kepentingan terbesar, diikuti `absences`, `failures`, `age`, dan beberapa faktor latar belakang. "
        "Anda dapat menguji perbandingan performa **Semua Variabel vs Fitur Penting Saja** pada halaman "
        "**Perbandingan Fitur** menggunakan tombol yang tersedia."
    )

with t2:
    st.write(
        f"Dataset aktif memiliki **{q['fitur']} fitur prediktor** ({q['numerik']} numerik dan {q['kategorikal']} kategorikal) "
        "serta 1 kolom target **`G3`** (nilai akhir 0–20)."
    )
    rows = []
    for col in df.columns:
        if col == "G3":
            rows.append({"Kolom": "G3", "Tipe": "Target (Numerik)", "Deskripsi": "Nilai akhir siswa (0–20)"})
        else:
            tipe, desc = DESKRIPSI_FITUR.get(col, ("Lainnya", "-"))
            rows.append({"Kolom": col, "Tipe": tipe, "Deskripsi": desc})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with t3:
    a, b, c, d = st.columns(4)
    a.metric("Baris (siswa)", q["baris"])
    b.metric("Total kolom", q["kolom"])
    c.metric("Fitur numerik", q["numerik"])
    d.metric("Fitur kategorikal", q["kategorikal"])

    pilih = st.radio(
        f"Filter berdasarkan threshold (G3 ≥ {threshold:.1f})",
        ["Semua", f"Lulus (G3 ≥ {threshold:.1f})", f"Berisiko (G3 < {threshold:.1f})"],
        horizontal=True,
    )
    n = st.slider("Jumlah baris ditampilkan", 5, 50, 10)
    if pilih.startswith("Lulus"):
        tampil = df[df["G3"] >= threshold]
    elif pilih.startswith("Berisiko"):
        tampil = df[df["G3"] < threshold]
    else:
        tampil = df
    st.dataframe(tampil.head(n), width="stretch", hide_index=True)

    with st.expander("Tipe data tiap kolom"):
        st.dataframe(df.dtypes.astype(str).rename("Tipe").to_frame(), width="stretch")
    with st.expander("Statistik deskriptif fitur numerik"):
        st.dataframe(df.describe().T.round(3), width="stretch")

with t4:
    cA, cB = st.columns([1.2, 1])
    status = np.where(df["G3"] >= threshold, f"Lulus (≥ {threshold:.1f})", f"Berisiko (< {threshold:.1f})")
    dist = pd.DataFrame({"Status": status}).value_counts().reset_index(name="Jumlah")
    fig_pie = px.pie(
        dist,
        names="Status",
        values="Jumlah",
        hole=0.5,
        color="Status",
        color_discrete_map={
            f"Lulus (≥ {threshold:.1f})": WARNA_AMAN,
            f"Berisiko (< {threshold:.1f})": WARNA_RISIKO,
        },
        title=f"Proporsi Lulus vs Berisiko (Threshold G3 = {threshold:.1f})",
    )
    cA.plotly_chart(tata_letak_plotly(fig_pie, 340), width="stretch")

    with cB:
        st.metric("Rata-rata G3", fmt(q["mean_g3"], 2))
        st.metric("Simpangan baku G3", fmt(q["std_g3"], 2))
        st.metric(f"Lulus (G3 ≥ {threshold:.1f})", f"{q['lulus']} ({fmt(q['rasio_lulus'], 1)}%)")
        st.metric(f"Berisiko (G3 < {threshold:.1f})", f"{q['risiko']} ({fmt(100 - q['rasio_lulus'], 1)}%)")

    fig_hist = px.histogram(
        df,
        x="G3",
        nbins=21,
        color="subject" if "subject" in df.columns else None,
        barmode="overlay",
        opacity=0.75,
        title="Distribusi Nilai Akhir (G3)",
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color=WARNA_RISIKO, annotation_text=f"Threshold {threshold:.1f}")
    st.plotly_chart(tata_letak_plotly(fig_hist, 350), width="stretch")
