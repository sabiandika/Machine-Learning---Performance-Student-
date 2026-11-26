import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from utils import (
    RANDOM_STATE,
    WARNA_AMAN,
    WARNA_RISIKO,
    ambil_pilihan_dataset,
    ambil_threshold,
    bagi_data,
    fmt,
    kotak,
    muat_data,
    ringkas_kualitas_data,
    siapkan_xy,
    tata_letak_plotly,
)

pilihan = ambil_pilihan_dataset()
threshold = ambil_threshold()

st.title("Eksplorasi Data (EDA)")
st.caption(f"Analisis menyeluruh kualitas data, distribusi, korelasi, dan 10 masalah kritis EDA pada **{pilihan}**")

df = muat_data(pilihan)
X, y = siapkan_xy(pilihan)
q = ringkas_kualitas_data(pilihan, threshold)
num_cols = X.select_dtypes(include=np.number).columns.tolist()

# ------------------------------------------------------------------ kualitas data
st.header("1. Pemeriksaan kualitas data mentah")
a, b, c, d = st.columns(4)
a.metric("Data kosong (missing)", q["missing"])
b.metric("Baris duplikat", q["duplikat"])
c.metric("Nilai G3 = 0", q["g3_nol"], help="Siswa tidak mengikuti ujian akhir atau mendapat nilai 0 (valid)")
d.metric("Outlier absences (IQR)", q["absences_outlier"], help=f"Batas atas IQR: {fmt(q['upper_iqr'], 1)} hari")

a2, b2, c2 = st.columns(3)
a2.metric("Fitur numerik miring", f"{q['miring']} dari {q['numerik']}", help="|skewness| > 1")
b2.metric("Rata-rata G3", fmt(q["mean_g3"], 2))
c2.metric(f"Rasio Lulus (≥ {threshold:.1f})", f"{fmt(q['rasio_lulus'], 1)}%")
st.caption("Nilai G3 = 0 dan absences tinggi adalah observasi nyata (bukan error input), sehingga tidak langsung dihapus.")

# ------------------------------------------------------------------ jelajah interaktif
st.header("2. Jelajah fitur, distribusi, & korelasi")
tabA, tabB, tabC = st.tabs(["Distribusi & Hubungan ke G3", "Heatmap & Peringkat Korelasi", "Perbandingan Subpopulasi"])

with tabA:
    fitur = st.selectbox("Pilih fitur untuk dieksplorasi", list(X.columns), index=list(X.columns).index("G2"))
    dd = df[[fitur, "G3"]].copy()
    dd["Status"] = np.where(dd["G3"] >= threshold, f"Lulus (≥ {threshold:.1f})", f"Berisiko (< {threshold:.1f})")

    c1, c2 = st.columns(2)
    if fitur in num_cols:
        f1 = px.histogram(
            dd,
            x=fitur,
            color="Status",
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={f"Lulus (≥ {threshold:.1f})": WARNA_AMAN, f"Berisiko (< {threshold:.1f})": WARNA_RISIKO},
            title=f"Distribusi {fitur} berdasarkan Status",
        )
        c1.plotly_chart(tata_letak_plotly(f1, 340), width="stretch")

        f2 = px.scatter(
            dd,
            x=fitur,
            y="G3",
            color="Status",
            opacity=0.7,
            color_discrete_map={f"Lulus (≥ {threshold:.1f})": WARNA_AMAN, f"Berisiko (< {threshold:.1f})": WARNA_RISIKO},
            title=f"Hubungan {fitur} vs G3 (Pearson r = {fmt(df[fitur].corr(df['G3']), 3)})",
        )
        f2.add_hline(y=threshold, line_dash="dash", line_color=WARNA_RISIKO)
        c2.plotly_chart(tata_letak_plotly(f2, 340), width="stretch")
        st.caption(f"Skewness {fitur}: **{fmt(df[fitur].skew(), 3)}** | Rentang: {df[fitur].min()} s.d. {df[fitur].max()}")
    else:
        f1 = px.box(
            dd,
            x=fitur,
            y="G3",
            color=fitur,
            title=f"Sebaran Nilai G3 berdasarkan {fitur}",
        )
        f1.add_hline(y=threshold, line_dash="dash", line_color=WARNA_RISIKO)
        c1.plotly_chart(tata_letak_plotly(f1, 350), width="stretch")

        rata = dd.groupby(fitur)["G3"].agg(["count", "mean", "median", "std"]).reset_index().round(2)
        c2.markdown(f"**Ringkasan G3 per kategori `{fitur}`**")
        c2.dataframe(rata, width="stretch", hide_index=True)

with tabB:
    corr_cols = num_cols + ["G3"]
    metode_corr = st.radio("Metode korelasi", ["Pearson (linear)", "Spearman (monotonik)"], horizontal=True)
    cm = df[corr_cols].corr(method="pearson" if "Pearson" in metode_corr else "spearman")
    fig_hm = px.imshow(cm, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title=f"Heatmap Korelasi ({metode_corr})")
    st.plotly_chart(tata_letak_plotly(fig_hm, 520), width="stretch")

    g3_corr = (
        cm["G3"]
        .drop("G3")
        .to_frame("Korelasi")
        .assign(Abs=lambda d: d["Korelasi"].abs())
        .sort_values("Abs", ascending=False)
        .drop(columns="Abs")
        .reset_index()
        .rename(columns={"index": "Fitur"})
    )
    fig_bar = px.bar(
        g3_corr.sort_values("Korelasi"),
        x="Korelasi",
        y="Fitur",
        orientation="h",
        color="Korelasi",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title=f"Peringkat Korelasi Fitur Numerik terhadap G3 ({metode_corr})",
    )
    st.plotly_chart(tata_letak_plotly(fig_bar, 460), width="stretch")
    st.caption(
        "Temuan utama: **G2** dan **G1** berkorelasi sangat kuat positif dengan **G3** (> 0,8), sedangkan **failures** "
        "berkorelasi negatif paling kuat. Fitur gaya hidup/demografi memiliki korelasi linear lebih lemah secara individual."
    )

with tabC:
    if "subject" in df.columns:
        fig_sub = px.box(
            df,
            x="subject",
            y="G3",
            color="subject",
            title="Perbandingan Distribusi Nilai G3: Matematika vs Bahasa Portugis",
        )
        st.plotly_chart(tata_letak_plotly(fig_sub, 360), width="stretch")
        st.dataframe(df.groupby("subject")[["G1", "G2", "G3", "absences", "failures"]].mean().round(2), width="stretch")
    else:
        fig_sch = px.box(df, x="school", y="G3", color="school", title="Perbandingan G3 berdasarkan Sekolah (GP vs MS)")
        st.plotly_chart(tata_letak_plotly(fig_sch, 360), width="stretch")

# ------------------------------------------------------------------ 10 masalah kritis EDA
st.header("3. 10 masalah kritis EDA & mitigasinya")
st.write(
    "Mengacu pada audit `Analisis_10_Masalah_Kritis_EDA_Student.ipynb`, setiap masalah didiagnosis langsung "
    "pada dataset aktif dan dibuktikan dengan perbandingan **Sebelum vs Sesudah**."
)


def sebelum_sesudah(sebelum, sesudah):
    c1, c2 = st.columns(2)
    with c1:
        kotak(sebelum, "sebelum", "SEBELUM")
    with c2:
        kotak(sesudah, "sesudah", "SESUDAH")


@st.cache_data
def hitung_10_masalah(pilihan_ds, th):
    d = muat_data(pilihan_ds)
    X_ds, y_ds = siapkan_xy(pilihan_ds)
    num_c = X_ds.select_dtypes(include=np.number).columns.tolist()
    X_num = X_ds[num_c].copy()
    r = {}

    # 1. Kebocoran data (Data Leakage) saat scaling sebelum split
    Xs = StandardScaler().fit_transform(X_num)
    a, b, c, d_y = train_test_split(Xs, y_ds, test_size=0.2, random_state=RANDOM_STATE)
    r["r2_bocor"] = float(r2_score(d_y, DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE).fit(a, c).predict(b)))
    Xtr, Xte, ytr, yte = train_test_split(X_num, y_ds, test_size=0.2, random_state=RANDOM_STATE)
    pipe_clean = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE))
    r["r2_pipa"] = float(r2_score(yte, pipe_clean.fit(Xtr, ytr).predict(Xte)))

    # 2. Skewness ekstrem pada absences
    r["skew_abs_raw"] = float(X_num["absences"].skew())
    r["skew_abs_std"] = float(pd.Series(StandardScaler().fit_transform(X_num[["absences"]]).ravel()).skew())
    r["skew_abs_log"] = float(np.log1p(X_num["absences"]).skew())

    # 3. Listwise deletion bias pada missing non-random (MNAR)
    rng = np.random.default_rng(RANDOM_STATE)
    w = np.where(y_ds < th, 3.0, 1.0)
    idx = rng.choice(len(X_num), max(10, int(0.10 * len(X_num))), replace=False, p=w / w.sum())
    Xm = X_num.copy()
    Xm.iloc[idx, Xm.columns.get_loc("G2")] = np.nan
    ok = Xm.notna().all(axis=1)
    r["n_drop"] = int(ok.sum())
    r["n_total"] = len(X_num)
    r["geser_mean"] = float(y_ds[ok].mean() - y_ds.mean())
    r["n_knn"] = KNNImputer(n_neighbors=5).fit_transform(StandardScaler().fit_transform(Xm)).shape[0]

    # 4. Multikolinearitas G1, G2, failures
    r["corr_g1_g2"] = float(X_num["G1"].corr(X_num["G2"]))

    # 5. Paradoks Akurasi jika dikonversi ke biner (Lulus vs Berisiko)
    y_bin = (y_ds < th).astype(int)
    dummy_pred = np.zeros(len(y_bin), dtype=int)
    r["dummy_acc"] = float(accuracy_score(y_bin, dummy_pred))
    r["dummy_f1"] = float(f1_score(y_bin, dummy_pred, average="macro"))
    r["risiko_pct"] = float(y_bin.mean() * 100)

    # 6. Bias linearitas Pearson vs Spearman & Mutual Information
    r["p_abs"] = float(stats.pearsonr(X_num["absences"], y_ds)[0])
    r["s_abs"] = float(stats.spearmanr(X_num["absences"], y_ds)[0])
    r["mi_abs"] = float(mutual_info_regression(X_num[["absences", "G2"]], y_ds, random_state=RANDOM_STATE)[0])

    # 7. Fitur ID / Kardinalitas Tinggi Sintetis
    X_id = X_num.copy()
    X_id["student_id_sintetis"] = np.arange(len(X_id))
    dt_id = DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE).fit(X_id, y_ds)
    r["imp_id"] = float(dict(zip(X_id.columns, dt_id.feature_importances_))["student_id_sintetis"])

    # 8. Masking Effect pada Z-Score klasik vs Modified Z-Score
    s = pd.concat([X_num["absences"], pd.Series([150.0, 200.0])], ignore_index=True)
    z = ((s - s.mean()) / s.std()).abs()
    med = s.median()
    mad = max(float((s - med).abs().median()), 1e-6)
    mz = (0.6745 * (s - med) / mad).abs()
    r["z_klasik"] = int((z > 3).sum())
    r["z_mod"] = int((mz > 3.5).sum())

    # 9. Multimodalitas / Subpopulasi (Matematika vs Portugis atau G3=0 vs G3>0)
    r["mean_g3_all"] = float(y_ds.mean())
    r["mean_g3_nonzero"] = float(y_ds[y_ds > 0].mean())
    r["n_g3_zero"] = int((y_ds == 0).sum())

    # 10. Outlier valid pada absences & Yeo-Johnson
    q1, q3 = X_num["absences"].quantile([0.25, 0.75])
    batas = q3 + 1.5 * (q3 - q1)
    out = X_num["absences"] > batas
    r["n_out"] = int(out.sum())
    r["mean_g3_out"] = float(y_ds[out].mean()) if out.any() else float(y_ds.mean())
    r["skew_yj"] = float(pd.Series(PowerTransformer("yeo-johnson").fit_transform(X_num[["absences"]]).ravel()).skew())
    return r


R = hitung_10_masalah(pilihan, threshold)

g1, g2, g3 = st.tabs([
    "Masalah 1–4: Kualitas & Preprocessing",
    "Masalah 5–7: Metrik, Korelasi, Identitas",
    "Masalah 8–10: Outlier & Subpopulasi",
])

with g1:
    st.subheader("1. Kebocoran data (Data Leakage)")
    sebelum_sesudah(
        f"Scaler di-fit <b>sebelum</b> split sehingga statistik data uji bocor ke train.<br>R² bocor: <b>{fmt(R['r2_bocor'], 4)}</b>.",
        f"Semua transformasi dibungkus <b>Pipeline</b> (fit hanya pada train).<br>R² aman: <b>{fmt(R['r2_pipa'], 4)}</b>. Isolasi tetap wajib untuk generalisasi.",
    )
    st.subheader("2. Skewness ekstrem pada absences")
    sebelum_sesudah(
        f"StandardScaler tidak memperbaiki kemencengan.<br>Skewness setelah StandardScaler: <b>{fmt(R['skew_abs_std'], 3)}</b> (asli {fmt(R['skew_abs_raw'], 3)}).",
        f"Transformasi <code>log1p</code> menekan kemencengan menjadi <b>{fmt(R['skew_abs_log'], 3)}</b>, dan model berbasis pohon (Random Forest) tahan terhadap skewness.",
    )
    st.subheader("3. Listwise deletion pada missing non-random")
    sebelum_sesudah(
        f"Menghapus baris kosong (simulasi MNAR) menyisakan {R['n_drop']} dari {R['n_total']} baris dan menggeser rata-rata G3 sebesar <b>{fmt(R['geser_mean'], 3)}</b>.",
        f"Imputasi dalam pipeline mempertahankan seluruh <b>{R['n_knn']}</b> baris tanpa membuang informasi siswa berisiko.",
    )
    st.subheader("4. Multikolinearitas tinggi (G1 & G2)")
    sebelum_sesudah(
        f"Korelasi Pearson antara <code>G1</code> dan <code>G2</code> mencapai <b>{fmt(R['corr_g1_g2'], 3)}</b>, membuat koefisien regresi linear biasa tidak stabil.",
        "Menggunakan <b>Random Forest / Extra Trees</b> atau <b>Ridge (L2)</b> yang tahan terhadap multikolinearitas.",
    )

with g2:
    st.subheader("5. Paradoks akurasi pada status kelulusan")
    sebelum_sesudah(
        f"Dengan threshold G3 = {threshold:.1f}, proporsi siswa berisiko adalah {fmt(R['risiko_pct'], 1)}%. Model dummy yang selalu menebak Lulus tetap mendapat Accuracy <b>{fmt(R['dummy_acc'], 3)}</b> tetapi Macro F1 hanya <b>{fmt(R['dummy_f1'], 3)}</b>.",
        "Evaluasi menggunakan <b>R², RMSE, MAE</b> untuk regresi, serta <b>Recall Risiko & F1</b> untuk klasifikasi berbasis threshold.",
    )
    st.subheader("6. Bias linearitas korelasi Pearson")
    sebelum_sesudah(
        f"Korelasi Pearson <code>absences</code> terhadap G3 hanya <b>{fmt(R['p_abs'], 3)}</b> sehingga tampak hampir tidak berpengaruh.",
        f"Spearman (<b>{fmt(R['s_abs'], 3)}</b>) dan Mutual Information (<b>{fmt(R['mi_abs'], 3)}</b>) menangkap pola non-linear, dan Random Forest tetap memanfaatkannya.",
    )
    st.subheader("7. Fitur identitas ber-kardinalitas tinggi")
    sebelum_sesudah(
        f"Jika kolom ID unik dimasukkan ke Decision Tree, ID dapat menyerap feature importance sebesar <b>{fmt(R['imp_id'] * 100, 2)}%</b> (overfitting).",
        "Dataset Student Performance bersih dari kolom ID; pastikan kolom identitas tidak pernah dimasukkan saat produksi.",
    )

with g3:
    st.subheader("8. Masking effect pada Z-Score klasik")
    sebelum_sesudah(
        f"Penambahan 2 nilai ekstrem pada <code>absences</code> menaikkan mean & std sehingga Z-Score klasik hanya mendeteksi <b>{R['z_klasik']}</b> outlier.",
        f"Modified Z-Score berbasis median & MAD tetap stabil dan mendeteksi <b>{R['z_mod']}</b> outlier.",
    )
    st.subheader("9. Multimodalitas & subpopulasi nilai nol (G3 = 0)")
    sebelum_sesudah(
        f"Rata-rata G3 keseluruhan (<b>{fmt(R['mean_g3_all'], 2)}</b>) tertarik turun oleh <b>{R['n_g3_zero']}</b> siswa dengan G3 = 0.",
        f"Rata-rata siswa dengan G3 > 0 adalah <b>{fmt(R['mean_g3_nonzero'], 2)}</b>. Model non-linear (Random Forest) memisahkan subpopulasi ini dengan baik.",
    )
    st.subheader("10. Outlier valid pada absences")
    sebelum_sesudah(
        f"Terdapat <b>{R['n_out']}</b> siswa dengan absensi di atas batas IQR (rata-rata G3 = <b>{fmt(R['mean_g3_out'], 2)}</b>). Menghapusnya berarti menghilangkan sinyal siswa berisiko.",
        f"Outlier valid tetap dipertahankan. Untuk model sensitif skala tersedia Yeo-Johnson (skewness turun ke <b>{fmt(R['skew_yj'], 3)}</b>).",
    )
