# Prediksi Performa Siswa (Student Performance)

Aplikasi Streamlit interaktif untuk eksplorasi data, analisis 10 masalah kritis EDA, preprocessing, perbandingan fitur (semua variabel vs fitur penting), pemodelan regresi (Random Forest, Extra Trees, Decision Tree, Ridge), evaluasi R², serta prediksi nilai akhir **G3 (0–20)** lengkap dengan **threshold kelulusan** yang dapat diatur.

## Cara Menjalankan

```bash
cd student_performance_app
streamlit run app.py
```

Lalu buka `http://localhost:8501` di browser.

## Fitur Utama
- **Pilihan Dataset**: Gabungan (Matematika + Portugis), Matematika saja (`student-mat.csv`), atau Portugis saja (`student-por.csv`).
- **Threshold Kelulusan (Sidebar)**: Mengatur batas nilai lulus `G3` secara dinamis beserta penjelasan trade-off Recall vs False Positive.
- **EDA Lengkap**: Kualitas data mentah, distribusi fitur, korelasi Pearson/Spearman, dan 10 masalah kritis EDA beserta bukti sebelum vs sesudah.
- **Perbandingan Variabel**: Tombol interaktif untuk membandingkan performa model dengan **Semua Variabel** vs **Fitur Penting Saja** (serta skenario tanpa G1 & G2).
- **Modelling & Evaluasi R²**: Metrik R², RMSE, MAE, Validasi Silang 5-Fold, Residual Plot, dan Confusion Matrix berbasis threshold.
