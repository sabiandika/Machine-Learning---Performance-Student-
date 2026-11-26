import plotly.express as px
import streamlit as st
from utils import (
    WARNA_AMAN,
    WARNA_NETRAL,
    WARNA_RISIKO,
    ambil_pilihan_dataset,
    bandingkan_skenario_fitur,
    fmt,
    hitung_kepentingan_fitur,
    kotak,
    tata_letak_plotly,
)

pilihan = ambil_pilihan_dataset()

st.title("Perbandingan Variabel & Seleksi Fitur")
st.caption(f"Apakah semua variabel perlu dipakai? Uji perbandingan pada **{pilihan}**")

df_imp = hitung_kepentingan_fitur(pilihan)

kotak(
    "<b>Jawaban singkat:</b> Model utama memakai seluruh variabel yang tersedia agar tidak kehilangan konteks, "
    "tetapi <b>tidak semua variabel memiliki bobot yang sama</b>. Nilai periode sebelumnya (<b>G2</b> dan <b>G1</b>) "
    "mendominasi prediksi, diikuti oleh <b>absences</b>, <b>failures</b>, dan beberapa faktor pendukung. "
    "Gunakan tombol di bawah untuk membandingkan performa <b>R²</b> secara langsung.",
    "info",
    "APAKAH SEMUA FITUR DIPAKAI?",
)

top_k = st.slider("Jumlah fitur penting (Top-K) yang dibandingkan", min_value=3, max_value=15, value=8, step=1)
df_banding, fitur_penting, _ = bandingkan_skenario_fitur(pilihan, top_k=top_k)

st.subheader("Pilih mode perbandingan (klik tombol)")
b1, b2, b3 = st.columns(3)
if "mode_fitur" not in st.session_state:
    st.session_state["mode_fitur"] = "semua_skenario"

if b1.button("Bandingkan Semua Variabel (Lengkap)", width="stretch", type="primary" if st.session_state["mode_fitur"] == "semua" else "secondary"):
    st.session_state["mode_fitur"] = "semua"
if b2.button(f"Fitur Penting Saja (Top-{top_k})", width="stretch", type="primary" if st.session_state["mode_fitur"] == "penting" else "secondary"):
    st.session_state["mode_fitur"] = "penting"
if b3.button("Bandingkan Semua Skenario (+ Tanpa G1/G2)", width="stretch", type="primary" if st.session_state["mode_fitur"] == "semua_skenario" else "secondary"):
    st.session_state["mode_fitur"] = "semua_skenario"

mode = st.session_state["mode_fitur"]

row_semua = df_banding.iloc[0]
row_penting = df_banding.iloc[1]
row_early = df_banding.iloc[2]

if mode == "semua":
    st.markdown(f"### Mode: Semua Variabel Lengkap ({int(row_semua['Jumlah Fitur'])} fitur)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R²", fmt(row_semua["Test R²"], 4))
    c2.metric("CV R² Mean", fmt(row_semua["CV R² Mean"], 4))
    c3.metric("Test RMSE", fmt(row_semua["Test RMSE"], 3))
    c4.metric("Test MAE", fmt(row_semua["Test MAE"], 3))
    st.write("**Seluruh variabel yang dipakai:**")
    st.code(row_semua["Daftar Fitur"], language="text")

elif mode == "penting":
    st.markdown(f"### Mode: Fitur Penting / Terpilih Saja (Top-{top_k} fitur)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R²", fmt(row_penting["Test R²"], 4), f"{row_penting['Test R²'] - row_semua['Test R²']:+.4f} vs Semua")
    c2.metric("CV R² Mean", fmt(row_penting["CV R² Mean"], 4), f"{row_penting['CV R² Mean'] - row_semua['CV R² Mean']:+.4f} vs Semua")
    c3.metric("Test RMSE", fmt(row_penting["Test RMSE"], 3), f"{row_penting['Test RMSE'] - row_semua['Test RMSE']:+.3f} vs Semua", delta_color="inverse")
    c4.metric("Test MAE", fmt(row_penting["Test MAE"], 3), f"{row_penting['Test MAE'] - row_semua['Test MAE']:+.3f} vs Semua", delta_color="inverse")
    st.write(f"**{top_k} fitur terpenting untuk kelangsungan model:**")
    st.code(row_penting["Daftar Fitur"], language="text")
    st.dataframe(df_imp.head(top_k)[["Variabel", "Importance", "Kumulatif (%)", "Tipe", "Deskripsi"]].round(4), width="stretch", hide_index=True)

else:
    st.markdown("### Perbandingan Lengkap: Semua Variabel vs Fitur Penting vs Tanpa G1 & G2")
    st.dataframe(
        df_banding[["Skenario", "Jumlah Fitur", "CV R² Mean", "Test R²", "Test RMSE", "Test MAE"]].style.format({
            "CV R² Mean": "{:.4f}",
            "Test R²": "{:.4f}",
            "Test RMSE": "{:.3f}",
            "Test MAE": "{:.3f}",
        }).highlight_max(subset=["CV R² Mean", "Test R²"], color="#b7e4c7").highlight_min(subset=["Test RMSE", "Test MAE"], color="#b7e4c7"),
        width="stretch",
        hide_index=True,
    )

    fig_cmp = px.bar(
        df_banding,
        x="Skenario",
        y="Test R²",
        color="Skenario",
        text=df_banding["Test R²"].map(lambda v: fmt(v, 4)),
        color_discrete_sequence=[WARNA_NETRAL, WARNA_AMAN, WARNA_RISIKO],
        title="Perbandingan Test R² Antar Skenario Fitur",
    )
    fig_cmp.update_layout(showlegend=False, yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(tata_letak_plotly(fig_cmp, 350), width="stretch")

    st.caption(
        f"Dengan hanya **{top_k} fitur terpenting**, Test R² mencapai **{fmt(row_penting['Test R²'], 4)}** "
        f"(dibanding **{fmt(row_semua['Test R²'], 4)}** saat memakai seluruh {int(row_semua['Jumlah Fitur'])} variabel). "
        f"Sebaliknya, jika `G1` dan `G2` dibuang (skenario early warning), Test R² turun tajam menjadi **{fmt(row_early['Test R²'], 4)}**."
    )

st.divider()
st.subheader("Peringkat Feature Importance Seluruh Variabel (Random Forest)")
fig_imp = px.bar(
    df_imp.head(15).sort_values("Importance"),
    x="Importance",
    y="Variabel",
    orientation="h",
    color="Importance",
    color_continuous_scale="Tealgrn",
    title="15 Variabel dengan Kontribusi Terbesar terhadap Prediksi G3",
)
fig_imp.update_layout(coloraxis_showscale=False)
st.plotly_chart(tata_letak_plotly(fig_imp, 420), width="stretch")

with st.expander("Lihat tabel kepentingan seluruh variabel"):
    st.dataframe(df_imp.round(4), width="stretch", hide_index=True)
