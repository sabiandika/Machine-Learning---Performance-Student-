"""
Fungsi bersama untuk aplikasi Prediksi Performa Siswa (Student Performance) - Streamlit.
Berisi: tema, pemuatan data (Matematika / Portugis / Gabungan), definisi model,
evaluasi R2/RMSE/MAE, perbandingan fitur, dan evaluasi berbasis threshold.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_DIR = BASE / "data"
MAT_PATH = DATA_DIR / "student-mat.csv" if (DATA_DIR / "student-mat.csv").exists() else ROOT / "student-mat.csv"
POR_PATH = DATA_DIR / "student-por.csv" if (DATA_DIR / "student-por.csv").exists() else ROOT / "student-por.csv"
ARTIFACT_PATH = (
    BASE / "model_artifacts" / "student_performance_rf.joblib"
    if (BASE / "model_artifacts" / "student_performance_rf.joblib").exists()
    else ROOT / "model_artifacts" / "student_performance_rf.joblib"
)
RANDOM_STATE = 42

WARNA_AMAN = "#2a9d8f"
WARNA_RISIKO = "#e63946"
WARNA_AKSEN = "#4361ee"
WARNA_NETRAL = "#6c757d"

PILIHAN_DATASET = [
    "Gabungan (Matematika + Portugis)",
    "Matematika (student-mat.csv)",
    "Portugis (student-por.csv)",
]

DESKRIPSI_FITUR = {
    "school": ("Kategorikal", "Sekolah siswa ('GP' = Gabriel Pereira, 'MS' = Mousinho da Silveira)"),
    "sex": ("Kategorikal", "Jenis kelamin ('F' = Perempuan, 'M' = Laki-laki)"),
    "age": ("Numerik", "Usia siswa (15–22 tahun)"),
    "address": ("Kategorikal", "Tipe tempat tinggal ('U' = Urban/Kota, 'R' = Rural/Desa)"),
    "famsize": ("Kategorikal", "Ukuran keluarga ('LE3' <= 3 orang, 'GT3' > 3 orang)"),
    "Pstatus": ("Kategorikal", "Status tinggal orang tua ('T' = Bersama, 'A' = Terpisah)"),
    "Medu": ("Numerik", "Pendidikan ibu (0: tidak sekolah s.d. 4: pendidikan tinggi)"),
    "Fedu": ("Numerik", "Pendidikan ayah (0: tidak sekolah s.d. 4: pendidikan tinggi)"),
    "Mjob": ("Kategorikal", "Pekerjaan ibu ('teacher', 'health', 'services', 'at_home', 'other')"),
    "Fjob": ("Kategorikal", "Pekerjaan ayah ('teacher', 'health', 'services', 'at_home', 'other')"),
    "reason": ("Kategorikal", "Alasan memilih sekolah ('home', 'reputation', 'course', 'other')"),
    "guardian": ("Kategorikal", "Wali siswa ('mother', 'father', 'other')"),
    "traveltime": ("Numerik", "Waktu tempuh rumah ke sekolah (1: <15 mnt s.d. 4: >1 jam)"),
    "studytime": ("Numerik", "Waktu belajar mingguan (1: <2 jam s.d. 4: >10 jam)"),
    "failures": ("Numerik", "Jumlah ketidaklulusan mata pelajaran sebelumnya (0–4)"),
    "schoolsup": ("Kategorikal", "Dukungan belajar tambahan dari sekolah ('yes'/'no')"),
    "famsup": ("Kategorikal", "Dukungan belajar dari keluarga ('yes'/'no')"),
    "paid": ("Kategorikal", "Mengikuti les tambahan berbayar ('yes'/'no')"),
    "activities": ("Kategorikal", "Aktivitas ekstrakurikuler ('yes'/'no')"),
    "nursery": ("Kategorikal", "Pernah mengikuti pendidikan prasekolah/TK ('yes'/'no')"),
    "higher": ("Kategorikal", "Berencana melanjutkan ke perguruan tinggi ('yes'/'no')"),
    "internet": ("Kategorikal", "Akses internet di rumah ('yes'/'no')"),
    "romantic": ("Kategorikal", "Sedang menjalin hubungan romantis ('yes'/'no')"),
    "famrel": ("Numerik", "Kualitas hubungan keluarga (1: sangat buruk s.d. 5: sangat baik)"),
    "freetime": ("Numerik", "Waktu luang setelah sekolah (1: sangat sedikit s.d. 5: sangat banyak)"),
    "goout": ("Numerik", "Frekuensi keluar bersama teman (1: sangat jarang s.d. 5: sangat sering)"),
    "Dalc": ("Numerik", "Konsumsi alkohol hari kerja (1: sangat rendah s.d. 5: sangat tinggi)"),
    "Walc": ("Numerik", "Konsumsi alkohol akhir pekan (1: sangat rendah s.d. 5: sangat tinggi)"),
    "health": ("Numerik", "Kondisi kesehatan saat ini (1: sangat buruk s.d. 5: sangat baik)"),
    "absences": ("Numerik", "Jumlah ketidakhadiran sekolah (0–93 hari)"),
    "G1": ("Numerik", "Nilai periode ke-1 (0–20)"),
    "G2": ("Numerik", "Nilai periode ke-2 (0–20)"),
    "subject": ("Kategorikal", "Mata pelajaran ('mathematics' atau 'portuguese') — ada pada dataset gabungan"),
}


# --------------------------------------------------------------------------- #
# Tema & Komponen UI
# --------------------------------------------------------------------------- #
def pasang_tema():
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
        h1 {{ letter-spacing: -0.5px; }}
        [data-testid="stMetric"] {{
            background: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.20);
            border-left: 5px solid {WARNA_AKSEN};
            padding: 0.8rem 1rem; border-radius: 10px;
        }}
        .kotak {{
            border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.5rem 0 1rem 0;
            border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
        }}
        .kotak.sebelum {{ border-left: 5px solid {WARNA_RISIKO}; }}
        .kotak.sesudah {{ border-left: 5px solid {WARNA_AMAN}; }}
        .kotak.info    {{ border-left: 5px solid {WARNA_AKSEN}; }}
        .kotak b.tag   {{ font-size: 0.75rem; letter-spacing: 1px; }}
        .hero {{
            padding: 2rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
            background: linear-gradient(135deg, {WARNA_AKSEN} 0%, #3a0ca3 100%); color: white;
        }}
        .hero h1 {{ color: white; margin: 0 0 .3rem 0; }}
        .hero p  {{ color: rgba(255,255,255,.92); margin: 0; font-size: 1.05rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(judul, subjudul):
    st.markdown(f'<div class="hero"><h1>{judul}</h1><p>{subjudul}</p></div>', unsafe_allow_html=True)


def kotak(isi, jenis="info", tag=None):
    label = f'<b class="tag">{tag}</b><br>' if tag else ""
    st.markdown(f'<div class="kotak {jenis}">{label}{isi}</div>', unsafe_allow_html=True)


def tata_letak_plotly(fig, tinggi=380):
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fmt(x, d=4):
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:.{d}f}".replace(".", ",")


def ambil_pilihan_dataset():
    return st.session_state.get("dataset_pilihan", PILIHAN_DATASET[0])


def ambil_threshold():
    return float(st.session_state.get("threshold_g3", 10.0))


# --------------------------------------------------------------------------- #
# Pemuatan Data
# --------------------------------------------------------------------------- #
@st.cache_data
def muat_data(pilihan=PILIHAN_DATASET[0]):
    mat = pd.read_csv(MAT_PATH, sep=";").assign(subject="mathematics")
    por = pd.read_csv(POR_PATH, sep=";").assign(subject="portuguese")
    if pilihan.startswith("Matematika"):
        return mat.drop(columns=["subject"]).reset_index(drop=True)
    if pilihan.startswith("Portugis"):
        return por.drop(columns=["subject"]).reset_index(drop=True)
    return pd.concat([mat, por], ignore_index=True)


@st.cache_data
def siapkan_xy(pilihan=PILIHAN_DATASET[0]):
    df = muat_data(pilihan)
    y = df["G3"].astype(float)
    X = df.drop(columns=["G3"])
    return X, y


@st.cache_data
def bagi_data(pilihan=PILIHAN_DATASET[0]):
    X, y = siapkan_xy(pilihan)
    grade_bin = pd.cut(y, bins=[-1, 9, 14, 20], labels=["low", "medium", "high"])
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=grade_bin)


@st.cache_data
def ringkas_kualitas_data(pilihan=PILIHAN_DATASET[0], threshold=10.0):
    df = muat_data(pilihan)
    X, y = siapkan_xy(pilihan)
    num_cols = X.select_dtypes(include=np.number).columns
    q1, q3 = df["absences"].quantile([0.25, 0.75])
    upper_iqr = q3 + 1.5 * (q3 - q1)
    skewed = (X[num_cols].skew().abs() > 1).sum()
    n_lulus = int((y >= threshold).sum())
    n_risiko = int((y < threshold).sum())
    return {
        "baris": len(df),
        "kolom": df.shape[1],
        "fitur": X.shape[1],
        "numerik": len(num_cols),
        "kategorikal": X.shape[1] - len(num_cols),
        "missing": int(df.isna().sum().sum()),
        "duplikat": int(df.duplicated().sum()),
        "g3_nol": int((df["G3"] == 0).sum()),
        "absences_outlier": int((df["absences"] > upper_iqr).sum()),
        "upper_iqr": float(upper_iqr),
        "miring": int(skewed),
        "lulus": n_lulus,
        "risiko": n_risiko,
        "rasio_lulus": n_lulus / len(df) * 100,
        "mean_g3": float(y.mean()),
        "std_g3": float(y.std()),
    }


# --------------------------------------------------------------------------- #
# Pipeline & Modelling
# --------------------------------------------------------------------------- #
def buat_preprocessor(X_sample, scale_numeric=False):
    num_cols = X_sample.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_sample.select_dtypes(exclude=np.number).columns.tolist()
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


NAMA_MODEL = [
    "Random Forest (Tuned)",
    "Extra Trees",
    "Ridge Regression",
    "Decision Tree (Baseline)",
]


def daftar_estimator():
    return {
        "Random Forest (Tuned)": (
            RandomForestRegressor(
                n_estimators=600,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features=0.7,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Extra Trees": (
            ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Ridge Regression": (
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree (Baseline)": (
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE),
            False,
        ),
    }


@st.cache_resource
def latih_semua_model(pilihan=PILIHAN_DATASET[0]):
    X_train, X_test, y_train, y_test = bagi_data(pilihan)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    prediksi = {}

    for nama, (est, perlu_scale) in daftar_estimator().items():
        pipe = Pipeline([
            ("preprocess", buat_preprocessor(X_train, scale_numeric=perlu_scale)),
            ("model", est),
        ])
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1,
            scoring={
                "r2": "r2",
                "neg_rmse": "neg_root_mean_squared_error",
                "neg_mae": "neg_mean_absolute_error",
            },
        )
        pipe.fit(X_train, y_train)
        y_pred = np.clip(pipe.predict(X_test), 0, 20)
        fitted[nama] = pipe
        prediksi[nama] = y_pred
        rows.append({
            "Model": nama,
            "CV R² Mean": float(cv_res["test_r2"].mean()),
            "CV R² Std": float(cv_res["test_r2"].std()),
            "CV RMSE": float(-cv_res["test_neg_rmse"].mean()),
            "CV MAE": float(-cv_res["test_neg_mae"].mean()),
            "Test R²": float(r2_score(y_test, y_pred)),
            "Test RMSE": float(root_mean_squared_error(y_test, y_pred)),
            "Test MAE": float(mean_absolute_error(y_test, y_pred)),
        })

    tabel = pd.DataFrame(rows).sort_values("Test R²", ascending=False).reset_index(drop=True)
    return tabel, fitted, prediksi


@st.cache_data
def hitung_kepentingan_fitur(pilihan=PILIHAN_DATASET[0]):
    """Menghitung feature importance dari Random Forest pada train set dan mengagregasi per variabel asal."""
    X_train, _, y_train, _ = bagi_data(pilihan)
    _, fitted, _ = latih_semua_model(pilihan)
    rf_pipe = fitted["Random Forest (Tuned)"]
    prep = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]
    feat_names = prep.get_feature_names_out()
    importances = model.feature_importances_

    # Agregasi one-hot kembali ke nama kolom asal
    asal_scores = {col: 0.0 for col in X_train.columns}
    for fname, imp in zip(feat_names, importances):
        matched = None
        for col in sorted(X_train.columns, key=len, reverse=True):
            if fname == col or fname.startswith(f"{col}_"):
                matched = col
                break
        if matched:
            asal_scores[matched] += float(imp)

    df_imp = (
        pd.DataFrame([{"Variabel": k, "Importance": v} for k, v in asal_scores.items()])
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    df_imp["Kumulatif (%)"] = df_imp["Importance"].cumsum() * 100
    df_imp["Tipe"] = df_imp["Variabel"].map(lambda c: DESKRIPSI_FITUR.get(c, ("Lainnya", ""))[0])
    df_imp["Deskripsi"] = df_imp["Variabel"].map(lambda c: DESKRIPSI_FITUR.get(c, ("", "-"))[1])
    return df_imp


@st.cache_data
def bandingkan_skenario_fitur(pilihan=PILIHAN_DATASET[0], top_k=8):
    """Membandingkan performa Random Forest pada: Semua Variabel vs Fitur Penting (Top-K) vs Tanpa G1 & G2."""
    X_train, X_test, y_train, y_test = bagi_data(pilihan)
    df_imp = hitung_kepentingan_fitur(pilihan)
    fitur_penting = df_imp["Variabel"].head(top_k).tolist()
    fitur_tanpa_g12 = [c for c in X_train.columns if c not in ("G1", "G2")]

    skenario = {
        f"Semua Variabel ({X_train.shape[1]} fitur)": list(X_train.columns),
        f"Fitur Penting Saja (Top-{top_k})": fitur_penting,
        f"Tanpa G1 & G2 / Early Warning ({len(fitur_tanpa_g12)} fitur)": fitur_tanpa_g12,
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    hasil = []
    pred_map = {}

    for label, cols in skenario.items():
        Xtr_sub = X_train[cols]
        Xte_sub = X_test[cols]
        pipe = Pipeline([
            ("preprocess", buat_preprocessor(Xtr_sub, scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features=0.7 if len(cols) > 3 else 1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])
        cv_res = cross_validate(
            pipe,
            Xtr_sub,
            y_train,
            cv=cv,
            n_jobs=-1,
            scoring={"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"},
        )
        pipe.fit(Xtr_sub, y_train)
        pred = np.clip(pipe.predict(Xte_sub), 0, 20)
        pred_map[label] = pred
        hasil.append({
            "Skenario": label,
            "Jumlah Fitur": len(cols),
            "CV R² Mean": float(cv_res["test_r2"].mean()),
            "Test R²": float(r2_score(y_test, pred)),
            "Test RMSE": float(root_mean_squared_error(y_test, pred)),
            "Test MAE": float(mean_absolute_error(y_test, pred)),
            "Daftar Fitur": ", ".join(cols),
        })

    return pd.DataFrame(hasil), fitur_penting, pred_map


def evaluasi_threshold(y_true, y_pred, threshold):
    """
    Mengubah prediksi regresi G3 menjadi klasifikasi biner berbasis threshold:
    Kelas Positif (1) = Berisiko Tidak Lulus (G3 < threshold)
    Kelas Negatif (0) = Lulus / Aman (G3 >= threshold)
    """
    aktual_risiko = (np.asarray(y_true) < threshold).astype(int)
    pred_risiko = (np.asarray(y_pred) < threshold).astype(int)
    tp = int(((aktual_risiko == 1) & (pred_risiko == 1)).sum())
    fn = int(((aktual_risiko == 1) & (pred_risiko == 0)).sum())
    fp = int(((aktual_risiko == 0) & (pred_risiko == 1)).sum())
    tn = int(((aktual_risiko == 0) & (pred_risiko == 0)).sum())
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Accuracy": float(accuracy_score(aktual_risiko, pred_risiko)),
        "Recall Risiko": float(recall_score(aktual_risiko, pred_risiko, zero_division=0)),
        "Precision Risiko": float(precision_score(aktual_risiko, pred_risiko, zero_division=0)),
        "F1 Risiko": float(f1_score(aktual_risiko, pred_risiko, zero_division=0)),
    }


@st.cache_resource
def muat_artefak_rf():
    if ARTIFACT_PATH.exists():
        return joblib.load(ARTIFACT_PATH)
    return None


def prediksi_dengan_interval(pipe, new_data):
    """Prediksi G3 beserta rentang ketidakpastian 80% antar-pohon pada Random Forest / Extra Trees."""
    point = np.clip(pipe.predict(new_data), 0, 20)
    model = pipe.named_steps["model"]
    if hasattr(model, "estimators_"):
        transformed = pipe.named_steps["preprocess"].transform(new_data)
        tree_preds = np.vstack([tree.predict(transformed) for tree in model.estimators_])
        lower, upper = np.quantile(tree_preds, [0.10, 0.90], axis=0)
        return (
            point,
            np.clip(lower, 0, 20),
            np.clip(upper, 0, 20),
        )
    return point, np.clip(point - 1.5, 0, 20), np.clip(point + 1.5, 0, 20)
