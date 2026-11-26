"""
Website Prediksi Performa Siswa (Student Performance) - Tugas Machine Learning.
Jalankan:  streamlit run app.py      lalu buka  http://localhost:8501
"""
import streamlit as st

st.set_page_config(
    page_title="Prediksi Performa Siswa (G3)",
    page_icon=":school:",
    layout="wide",
)

from utils import PILIHAN_DATASET, ambil_pilihan_dataset, ambil_threshold, pasang_tema  # noqa: E402

pasang_tema()

halaman = {
    "Pengantar": [
        st.Page("halaman/beranda.py", title="Beranda", icon=":material/home:", default=True),
        st.Page("halaman/dataset.py", title="Tentang Dataset", icon=":material/database:"),
        st.Page("halaman/eda.py", title="Eksplorasi Data (EDA)", icon=":material/bar_chart:"),
    ],
    "Pengolahan & Model": [
        st.Page("halaman/preprocessing.py", title="Preprocessing", icon=":material/tune:"),
        st.Page("halaman/seleksi_fitur.py", title="Perbandingan Fitur", icon=":material/compare_arrows:"),
        st.Page("halaman/modelling.py", title="Modelling & Evaluasi (R²)", icon=":material/model_training:"),
    ],
    "Penerapan": [
        st.Page("halaman/prediksi.py", title="Prediksi Satu Siswa", icon=":material/person_search:"),
        st.Page("halaman/massal.py", title="Prediksi Massal (CSV)", icon=":material/upload_file:"),
        st.Page("halaman/kesimpulan.py", title="Kesimpulan & Saran", icon=":material/task_alt:"),
    ],
}

nav = st.navigation(halaman)

with st.sidebar:
    st.markdown("### Pengaturan Global")
    st.selectbox(
        "Pilih Dataset",
        PILIHAN_DATASET,
        index=0,
        key="dataset_pilihan",
        help="Ganti antara data gabungan, Matematika saja, atau Bahasa Portugis saja.",
    )

    st.slider(
        "Threshold kelulusan G3",
        min_value=4.0,
        max_value=16.0,
        value=10.0,
        step=0.5,
        key="threshold_g3",
    )
    st.caption(
        "Threshold lebih tinggi membuat standar kelulusan lebih ketat (Recall siswa berisiko naik), "
        "tetapi dapat menambah False Positive (siswa aman ikut tertandai berisiko). "
        "Threshold lebih rendah melonggarkan standar sehingga risiko False Negative meningkat."
    )

    st.divider()
    st.markdown("**Konfigurasi model**")
    st.caption(
        "Metode utama: Random Forest Regressor "
        "(600 pohon, max_features=0.7, min_samples_leaf=2)\n\n"
        "Pembanding: Extra Trees, Ridge Regression, Decision Tree\n\n"
        "Preprocessing: ColumnTransformer (Median/Modus + OneHotEncoder)\n\n"
        "Pembagian data: 80% training dan 20% testing (stratified)"
    )
    st.divider()
    st.caption(f"Dataset aktif: **{ambil_pilihan_dataset()}**  \nThreshold aktif: **G3 ≥ {ambil_threshold():.1f}**")

nav.run()
