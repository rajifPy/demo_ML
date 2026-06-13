# -*- coding: utf-8 -*-
"""
=============================================================================
  PREPROCESSING PIPELINE (v2)
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
  Kelompok E – Machine Learning SD-A1, Universitas Airlangga
=============================================================================

Dataset yang digunakan (13 dataset):
  1.  Indeks Harga Konsumen (Umum)                    – BPS, 2005–2019
  2.  Inflasi Bulanan (M-to-M)                        – BPS, 2005–2026
  3.  Tingkat Inflasi Tahun Kalender (Y-to-D)         – BPS, referensi
  4.  BI Rate / Data Inflasi BI                       – Bank Indonesia
  5.  Upah Minimum Provinsi (UMP)                     – BPS Jateng, 2021–2025
  6.  Rata-rata Pengeluaran per Kapita                 – BPS, 2017–2025
  7.  Data Historis USD/IDR                           – Investing.com, bulanan
  8.  Tingkat Pengangguran Terbuka (Semester+Provinsi) – Open Data Jabar
  9.  TPT & TPAK Menurut Provinsi                     – BPS, 2017–2025
  10. PDRB Per Kapita (Ribu Rupiah)                   – BPS, 2010–2025
  11. Persentase Penduduk Miskin per Provinsi          – BPS, 2010–2024
  12. Inflasi Umum, Inti, Harga Diatur, Bergejolak    – BPS, 2009–2026
  13. Harga Bulanan Minyak Mentah (USD/Barel)          – IndexMundi, 2001–2026

Output:
  1. datasets/processed/clean_inflasi_ts.csv  → Model 1 (LSTM Forecasting)
  2. datasets/processed/clean_daya_beli.csv   → Model 2 (Regresi Daya Beli)
=============================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

BASE = "datasets"
OUT_DIR = os.path.join(BASE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BULAN_MAP = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12,
}

BULAN_MAP_EN = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_indo_date(s: str) -> pd.Timestamp:
    """Parse tanggal format 'Januari 2024' atau 'Jan 2024'."""
    try:
        parts = str(s).strip().split()
        if len(parts) == 2:
            bulan = BULAN_MAP.get(parts[0]) or BULAN_MAP_EN.get(parts[0])
            if bulan:
                return pd.Timestamp(year=int(parts[1]), month=bulan, day=1)
    except Exception:
        pass
    return pd.NaT


def _to_float_id(val) -> float:
    """Konversi angka format Indonesia (1.234,56 → 1234.56) ke float."""
    try:
        s = str(val).strip()
        if s in ("-", "", "nan", "None", "-"):
            return np.nan
        s = s.replace("%", "").replace(" ", "")
        # Format Indonesia: titik = ribuan, koma = desimal
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "." in s and s.count(".") > 1:
            s = s.replace(".", "")
        return float(s)
    except Exception:
        return np.nan


def _extract_year(filename: str):
    """Ekstrak tahun dari nama file (misalnya '...2024.csv' → 2024)."""
    try:
        stem = os.path.splitext(os.path.basename(filename))[0]
        # Handle '(1)' suffix: ambil token numerik 4-digit terakhir
        for part in reversed(stem.replace("(", " ").replace(")", " ").split()):
            if part.isdigit() and len(part) == 4:
                return int(part)
    except Exception:
        pass
    return None


def _find_indonesia(df: pd.DataFrame, col: int = 0):
    """Cari baris dengan nilai 'INDONESIA' di kolom tertentu."""
    mask = df.iloc[:, col].astype(str).str.upper().str.strip() == "INDONESIA"
    return df[mask].iloc[0] if mask.any() else None


def _normalize_prov(name: str) -> str:
    """Normalisasi nama provinsi ke Title Case standar."""
    mapping = {
        "DKI JAKARTA": "DKI Jakarta",
        "DI YOGYAKARTA": "DI Yogyakarta",
        "ACEH": "Aceh",
        "KEP. BANGKA BELITUNG": "Kepulauan Bangka Belitung",
        "KEP. RIAU": "Kepulauan Riau",
        "KEPULAUAN BANGKA BELITUNG": "Kepulauan Bangka Belitung",
        "KEPULAUAN RIAU": "Kepulauan Riau",
    }
    u = str(name).strip().upper()
    if u in mapping:
        return mapping[u]
    return str(name).strip().title()


# ===========================================================================
# LOADERS
# ===========================================================================

# ---------------------------------------------------------------------------
# [1] Inflasi Bulanan (M-to-M) — backbone time series
# ---------------------------------------------------------------------------
def load_inflasi_mom() -> pd.DataFrame:
    """Inflasi Bulanan M-to-M → Series bulanan level INDONESIA."""
    print("  [1/13] Inflasi Bulanan M-to-M...", end=" ")
    files = glob.glob(os.path.join(BASE, "Inflasi Bulanan", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia(df)
            if row is None:
                continue
            for nama, angka in BULAN_MAP.items():
                if nama in df.columns:
                    val = _to_float_id(row[nama])
                    if not np.isnan(val):
                        records.append({"Tanggal": pd.Timestamp(tahun, angka, 1),
                                        "Inflasi_MoM": val})
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


# ---------------------------------------------------------------------------
# [2] Indeks Harga Konsumen (IHK) — level nasional
# ---------------------------------------------------------------------------
def load_ihk() -> pd.DataFrame:
    """IHK Nasional (Umum) → Series bulanan."""
    print("  [2/13] Indeks Harga Konsumen (IHK)...", end=" ")
    files = glob.glob(os.path.join(BASE, "Indeks Harga Konsumen (Umum)", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia(df)
            if row is None:
                continue
            for nama, angka in BULAN_MAP.items():
                if nama in df.columns:
                    val = _to_float_id(row[nama])
                    if not np.isnan(val):
                        records.append({"Tanggal": pd.Timestamp(tahun, angka, 1),
                                        "IHK": val})
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


# ---------------------------------------------------------------------------
# [4] BI Rate — suku bunga acuan
# ---------------------------------------------------------------------------
def load_bi_rate() -> pd.DataFrame:
    """BI Rate / Data Inflasi BI → Series bulanan."""
    print("  [4/13] BI Rate (Data Inflasi BI)...", end=" ")
    path = os.path.join(BASE, "BI Rate (Data Inflasi)", "Data Inflasi.xlsx")
    try:
        df = pd.read_excel(path, skiprows=3, header=0)
        if "Periode" not in df.columns:
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        df = df[["Periode", "Data Inflasi"]].dropna()
        df["Data Inflasi"] = df["Data Inflasi"].apply(_to_float_id)
        df["Tanggal"] = df["Periode"].apply(_parse_indo_date)
        df = (df.dropna(subset=["Tanggal", "Data Inflasi"])
              .sort_values("Tanggal")
              .set_index("Tanggal")
              [["Data Inflasi"]]
              .rename(columns={"Data Inflasi": "BI_Rate"}))
        print(f"{len(df)} baris ({df.index.min().year}–{df.index.max().year})")
        return df
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [7] Kurs USD/IDR — bulanan
# ---------------------------------------------------------------------------
def load_usd_idr() -> pd.DataFrame:
    """USD/IDR kurs bulanan dari Investing.com."""
    print("  [7/13] Kurs USD/IDR (bulanan)...", end=" ")
    folder = os.path.join(BASE, "Data Historis USD_IDR")
    # Cari file CSV apapun dalam folder (nama bisa berubah)
    candidates = glob.glob(os.path.join(folder, "*.csv"))
    if not candidates:
        print("GAGAL – tidak ada file CSV")
        return pd.DataFrame()
    path = candidates[0]
    try:
        df = pd.read_csv(path, dtype=str)
        # Rename kolom pertama & kedua
        df.rename(columns={df.columns[0]: "Tanggal", df.columns[1]: "Kurs"}, inplace=True)
        # Parse tanggal format dd/mm/yyyy
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
        df["Kurs"] = df["Kurs"].apply(_to_float_id)
        df = df.dropna(subset=["Tanggal", "Kurs"]).set_index("Tanggal").sort_index()
        # Data sudah bulanan - normalize ke awal bulan
        df.index = df.index.normalize()  # set to start of day
        # Resample ke frekuensi bulanan, ambil rata-rata
        df_monthly = df[["Kurs"]].resample("MS").mean().rename(columns={"Kurs": "USD_IDR"})
        print(f"{len(df_monthly)} bulan ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [12] Inflasi Umum, Inti, Harga Diatur, Bergejolak — komponen inflasi bulanan
# ---------------------------------------------------------------------------
def load_inflasi_komponen() -> pd.DataFrame:
    """
    Inflasi Umum, Inti, Harga Diatur Pemerintah, dan Bergejolak (M-to-M).
    Format: hierarkis Tahun → Bulan dengan 4 kolom nilai.
    """
    print("  [12/13] Inflasi Komponen (Umum/Inti/Diatur/Bergejolak)...", end=" ")
    folder = "Inflasi Umum, Inti, Harga Diatur Pemerintah, dan Bergejolak Nasional (M-to-M dan Y-to-D)"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    if not files:
        print("GAGAL – file tidak ditemukan")
        return pd.DataFrame()
    records = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, header=None, dtype=str, on_bad_lines="skip")
            current_year = None
            for _, row in df.iterrows():
                col0 = str(row.iloc[0]).strip() if len(row) > 0 else ""
                col1 = str(row.iloc[1]).strip() if len(row) > 1 else ""
                # Baris tahun: kolom pertama adalah angka 4-digit, kolom kedua kosong/NaN
                if col0.isdigit() and len(col0) == 4:
                    current_year = int(col0)
                    continue
                # Baris bulan: kolom pertama kosong, kolom kedua adalah nama bulan Indonesia
                if col0 in ("", "nan") and col1 in BULAN_MAP and current_year is not None:
                    bulan = BULAN_MAP[col1]
                    vals = []
                    for i in [2, 3, 4, 5]:
                        v = _to_float_id(row.iloc[i]) if len(row) > i else np.nan
                        vals.append(v)
                    records.append({
                        "Tanggal": pd.Timestamp(current_year, bulan, 1),
                        "Inflasi_Umum_MoM": vals[0],
                        "Inflasi_Inti_MoM": vals[1],
                        "Inflasi_HargaDiatur_MoM": vals[2],
                        "Inflasi_Bergejolak_MoM": vals[3],
                    })
        except Exception as err:
            print(f"  (parse error: {err})")
            pass
    if not records:
        print("GAGAL – tidak ada data terparsing")
        return pd.DataFrame()
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


# ---------------------------------------------------------------------------
# [13] Harga Minyak Mentah — bulanan USD/barel
# ---------------------------------------------------------------------------
def load_harga_minyak() -> pd.DataFrame:
    """Harga Bulanan Minyak Mentah (USD/Barel) dari IndexMundi."""
    print("  [13/13] Harga Minyak Mentah (USD/Barel)...", end=" ")
    folder = "Harga Bulanan Minyak Mentah (minyak bumi) - Dolar AS per Barel"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    if not files:
        print("GAGAL – file tidak ditemukan")
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[0], dtype=str)
        # Kolom: date, month_label, crude_oil_price_usd_per_barrel, monthly_change_percent
        df["Tanggal"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["Harga_Minyak_USD"] = df["crude_oil_price_usd_per_barrel"].apply(_to_float_id)
        df = (df.dropna(subset=["Tanggal", "Harga_Minyak_USD"])
              .sort_values("Tanggal")
              .set_index("Tanggal")
              [["Harga_Minyak_USD"]])
        # Normalize ke awal bulan dan resample
        df_monthly = df.resample("MS").mean()
        print(f"{len(df_monthly)} bulan ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [5] Upah Minimum Provinsi (UMP)
# ---------------------------------------------------------------------------
def load_ump() -> pd.DataFrame:
    """UMP per Provinsi per Tahun."""
    print("  [5/13] Upah Minimum Provinsi (UMP)...", end=" ")
    files = glob.glob(os.path.join(BASE, "Upah Minimum Provinsi", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=2, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi", df.columns[1]: "UMP"}, inplace=True)
            df = df.dropna(subset=["Provinsi", "UMP"])
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "INDONESIA", "NASIONAL", ""])]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            df["UMP"] = df["UMP"].apply(_to_float_id)
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "UMP", "Tahun"]].dropna())
        except Exception:
            pass
    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun")
    return df_out


# ---------------------------------------------------------------------------
# [6] Rata-rata Pengeluaran per Kapita
# ---------------------------------------------------------------------------
def load_pengeluaran() -> pd.DataFrame:
    """Rata-rata Pengeluaran per Kapita per Provinsi per Tahun."""
    print("  [6/13] Pengeluaran per Kapita...", end=" ")
    folder = "Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "", "INDONESIA"])]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            if len(df.columns) >= 4:
                col_makanan = df.columns[1]
                col_bukan = df.columns[2]
                col_total = df.columns[3]
                df["Pengeluaran_Makanan"] = df[col_makanan].apply(_to_float_id)
                df["Pengeluaran_Bukan_Makanan"] = df[col_bukan].apply(_to_float_id)
                df["Total_Pengeluaran"] = df[col_total].apply(_to_float_id)
            elif len(df.columns) == 3:
                # Beberapa file mungkin hanya punya 3 kolom
                col_makanan = df.columns[1]
                col_bukan = df.columns[2]
                df["Pengeluaran_Makanan"] = df[col_makanan].apply(_to_float_id)
                df["Pengeluaran_Bukan_Makanan"] = df[col_bukan].apply(_to_float_id)
                df["Total_Pengeluaran"] = df["Pengeluaran_Makanan"] + df["Pengeluaran_Bukan_Makanan"]
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "Pengeluaran_Makanan",
                                "Pengeluaran_Bukan_Makanan", "Total_Pengeluaran",
                                "Tahun"]].dropna())
        except Exception:
            pass
    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun")
    return df_out


# ---------------------------------------------------------------------------
# [8] Tingkat Pengangguran Terbuka — Semester & Provinsi (Open Data Jabar)
# ---------------------------------------------------------------------------
def load_pengangguran_semester() -> pd.DataFrame:
    """TPT per Provinsi per Tahun (rata-rata Feb+Agustus) — Open Data Jabar."""
    print("  [8/13] TPT Semester & Provinsi (Open Data Jabar)...", end=" ")
    path = os.path.join(
        BASE,
        "Tingkat Pengangguran Terbuka Berdasarkan Semester dan Provinsi di Indonesia",
        "disnakertrans-od_21012_tingkat_pengangguran_terbuka_brdsrkn_semester_prov_v1_data.csv"
    )
    try:
        df = pd.read_csv(path, dtype=str)
        df["tingkat_pengangguran_terbuka"] = df["tingkat_pengangguran_terbuka"].apply(_to_float_id)
        df["tahun"] = df["tahun"].apply(lambda x: int(x) if str(x).isdigit() else np.nan)
        df["nama_provinsi"] = df["nama_provinsi"].apply(_normalize_prov)
        agg = (df.dropna(subset=["tingkat_pengangguran_terbuka", "tahun"])
               .groupby(["nama_provinsi", "tahun"])["tingkat_pengangguran_terbuka"]
               .mean()
               .reset_index()
               .rename(columns={"nama_provinsi": "Provinsi",
                                "tahun": "Tahun",
                                "tingkat_pengangguran_terbuka": "TPT"}))
        agg["Tahun"] = agg["Tahun"].astype(int)
        print(f"{len(agg)} baris, {agg['Tahun'].nunique()} tahun "
              f"({int(agg['Tahun'].min())}–{int(agg['Tahun'].max())})")
        return agg
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [9] TPT & TPAK Menurut Provinsi — BPS per tahun
# ---------------------------------------------------------------------------
def load_tpt_tpak() -> pd.DataFrame:
    """
    TPT & TPAK per Provinsi per Tahun dari BPS.
    Rata-rata Feb + Agustus untuk TPT; rata-rata Feb + Agustus untuk TPAK.
    """
    print("  [9/13] TPT & TPAK Menurut Provinsi (BPS)...", end=" ")
    folder = "Tingkat Pengangguran Terbuka (TPT) dan Tingkat Partisipasi Angkatan Kerja (TPAK) Menurut Provinsi"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            # Filter baris meta/catatan
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "", "INDONESIA", "CATATAN"])]
            df = df[~df["Provinsi"].str.startswith("<sup", na=False)]
            df = df[~df["Provinsi"].str.startswith("Catatan", na=False)]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            # Cari kolom TPT dan TPAK
            cols = df.columns.tolist()
            tpt_cols = [c for c in cols if "TPT" in str(c).upper() or "Pengangguran" in str(c)]
            tpak_cols = [c for c in cols if "TPAK" in str(c).upper() or "Partisipasi" in str(c)]
            # Hitung rata-rata dari semua kolom TPT dan TPAK
            for _, row in df.iterrows():
                prov = row["Provinsi"]
                tpt_vals = [_to_float_id(row[c]) for c in tpt_cols if c in row]
                tpak_vals = [_to_float_id(row[c]) for c in tpak_cols if c in row]
                tpt_mean = np.nanmean(tpt_vals) if tpt_vals else np.nan
                tpak_mean = np.nanmean(tpak_vals) if tpak_vals else np.nan
                if not np.isnan(tpt_mean) or not np.isnan(tpak_mean):
                    records.append({
                        "Provinsi": prov,
                        "Tahun": tahun,
                        "TPT_BPS": round(tpt_mean, 4) if not np.isnan(tpt_mean) else np.nan,
                        "TPAK_BPS": round(tpak_mean, 4) if not np.isnan(tpak_mean) else np.nan,
                    })
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .dropna(subset=["Provinsi"])
              .drop_duplicates(["Provinsi", "Tahun"])
              .sort_values(["Tahun", "Provinsi"])
              .reset_index(drop=True))
    if not df_out.empty:
        print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun "
              f"({int(df_out['Tahun'].min())}–{int(df_out['Tahun'].max())})")
    else:
        print("GAGAL – tidak ada data")
    return df_out


# ---------------------------------------------------------------------------
# [10] PDRB Per Kapita (Ribu Rupiah) — BPS per tahun
# ---------------------------------------------------------------------------
def load_pdrb() -> pd.DataFrame:
    """PDRB Per Kapita per Provinsi per Tahun (Harga Berlaku & Konstan)."""
    print("  [10/13] PDRB Per Kapita (Ribu Rp)...", end=" ")
    folder = "Produk Domestik Regional Bruto Per Kapita (Ribu Rupiah)"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=4, header=None, dtype=str, on_bad_lines="skip")
            if df.empty or len(df.columns) < 2:
                continue
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            df = df[df["Provinsi"].str.strip() != ""]
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["INDONESIA", ""])]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            # Kolom 1 = Harga Berlaku, Kolom 2 = Harga Konstan 2010
            col_berlaku = df.columns[1] if len(df.columns) > 1 else None
            col_konstan = df.columns[2] if len(df.columns) > 2 else None
            for _, row in df.iterrows():
                prov = row["Provinsi"]
                berlaku = _to_float_id(row[col_berlaku]) if col_berlaku else np.nan
                konstan = _to_float_id(row[col_konstan]) if col_konstan else np.nan
                if not np.isnan(berlaku) or not np.isnan(konstan):
                    records.append({
                        "Provinsi": prov,
                        "Tahun": tahun,
                        "PDRB_HargaBerlaku": berlaku,
                        "PDRB_HargaKonstan": konstan,
                    })
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .dropna(subset=["Provinsi"])
              .drop_duplicates(["Provinsi", "Tahun"])
              .sort_values(["Tahun", "Provinsi"])
              .reset_index(drop=True))
    if not df_out.empty:
        print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun "
              f"({int(df_out['Tahun'].min())}–{int(df_out['Tahun'].max())})")
    else:
        print("GAGAL – tidak ada data")
    return df_out


# ---------------------------------------------------------------------------
# [11] Persentase Penduduk Miskin per Provinsi
# ---------------------------------------------------------------------------
def load_penduduk_miskin() -> pd.DataFrame:
    """Persentase Penduduk Miskin per Provinsi per Tahun."""
    print("  [11/13] Persentase Penduduk Miskin...", end=" ")
    folder = "Persentase Penduduk Miskin Berdasarkan Provinsi di Indonesia"
    files = glob.glob(os.path.join(BASE, folder, "*_data.csv"))
    if not files:
        files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    if not files:
        print("GAGAL – file tidak ditemukan")
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[0], dtype=str)
        # Kolom: id, kode_provinsi, nama_provinsi, persentase_penduduk_miskin, satuan, tahun
        df["persentase_penduduk_miskin"] = df["persentase_penduduk_miskin"].apply(_to_float_id)
        df["tahun"] = df["tahun"].apply(lambda x: int(x) if str(x).strip().isdigit() else np.nan)
        df["nama_provinsi"] = df["nama_provinsi"].apply(_normalize_prov)
        # Filter nilai 0.0 yang mengindikasikan data tidak tersedia (provinsi baru)
        df = df[df["persentase_penduduk_miskin"] > 0]
        df_out = (df.dropna(subset=["persentase_penduduk_miskin", "tahun"])
                  .rename(columns={
                      "nama_provinsi": "Provinsi",
                      "tahun": "Tahun",
                      "persentase_penduduk_miskin": "Pct_Penduduk_Miskin"
                  })
                  [["Provinsi", "Tahun", "Pct_Penduduk_Miskin"]]
                  .drop_duplicates(["Provinsi", "Tahun"])
                  .sort_values(["Tahun", "Provinsi"])
                  .reset_index(drop=True))
        df_out["Tahun"] = df_out["Tahun"].astype(int)
        print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun "
              f"({int(df_out['Tahun'].min())}–{int(df_out['Tahun'].max())})")
        return df_out
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ===========================================================================
# BUILD OUTPUT 1: clean_inflasi_ts.csv
# Time-series bulanan untuk Model 1 (LSTM Forecasting)
# ===========================================================================

def build_inflasi_ts(inflasi, ihk, bi_rate, usd_idr,
                     inflasi_komp, harga_minyak) -> pd.DataFrame:
    """
    Gabungkan semua fitur time-series bulanan:
    - Inflasi MoM (backbone, target)
    - IHK (hanya 2005–2019)
    - BI Rate
    - USD/IDR
    - Inflasi Komponen (Inti, Harga Diatur, Bergejolak)
    - Harga Minyak Mentah (USD/Barel)
    """
    print("\n▶ Membangun clean_inflasi_ts.csv ...")

    ts = inflasi.copy()

    # Merge IHK (hanya 2005–2019, NaN setelah itu)
    if not ihk.empty:
        ts = ts.join(ihk, how="left")

    # Merge BI Rate
    if not bi_rate.empty:
        ts = ts.join(bi_rate, how="left")

    # Merge USD/IDR
    if not usd_idr.empty:
        ts = ts.join(usd_idr, how="left")

    # Merge Inflasi Komponen (Inti, Harga Diatur, Bergejolak, Umum)
    if not inflasi_komp.empty:
        ts = ts.join(inflasi_komp, how="left")

    # Merge Harga Minyak Mentah
    if not harga_minyak.empty:
        ts = ts.join(harga_minyak, how="left")

    # Tambahkan fitur waktu (aman dari leakage)
    ts["Bulan"] = ts.index.month
    ts["Tahun"] = ts.index.year

    # Reset index agar Tanggal menjadi kolom biasa
    ts = ts.reset_index()

    out_path = os.path.join(OUT_DIR, "clean_inflasi_ts.csv")
    ts.to_csv(out_path, index=False)

    print(f"   ✓ {len(ts)} baris × {len(ts.columns)} kolom")
    print(f"   ✓ Rentang: {ts['Tanggal'].min().strftime('%b %Y')} – {ts['Tanggal'].max().strftime('%b %Y')}")
    print(f"   ✓ Kolom: {list(ts.columns)}")
    print(f"   ✓ Disimpan → {out_path}")
    return ts


# ===========================================================================
# BUILD OUTPUT 2: clean_daya_beli.csv
# Panel data provinsi untuk Model 2 (Regresi Daya Beli)
# ===========================================================================

def build_daya_beli_panel(inflasi, ump, pengeluaran,
                          pengangguran_sem, tpt_tpak,
                          pdrb, penduduk_miskin) -> pd.DataFrame:
    """
    Panel data provinsi × tahun:
    - Pengeluaran per Kapita (target Y)
    - UMP
    - TPT (dari TPT-BPS, fallback ke Semester)
    - TPAK
    - PDRB per Kapita
    - Persentase Penduduk Miskin
    - Inflasi rata-rata tahunan (dari Inflasi MoM)
    """
    print("\n▶ Membangun clean_daya_beli.csv ...")

    # --- Inflasi → rata-rata tahunan ---
    inflasi_tahunan = (inflasi.reset_index()
                       .assign(Tahun=lambda x: x["Tanggal"].dt.year)
                       .groupby("Tahun")["Inflasi_MoM"]
                       .mean()
                       .reset_index()
                       .rename(columns={"Inflasi_MoM": "Inflasi_Rata_Tahunan"}))

    # --- Normalisasi nama provinsi di semua dataset ---
    def norm_prov(df, col="Provinsi"):
        df = df.copy()
        df[col] = df[col].apply(_normalize_prov)
        return df

    pen_c = norm_prov(pengeluaran)
    ump_c = norm_prov(ump)
    tpt_sem_c = norm_prov(pengangguran_sem) if not pengangguran_sem.empty else pd.DataFrame()
    tpt_bps_c = norm_prov(tpt_tpak) if not tpt_tpak.empty else pd.DataFrame()
    pdrb_c = norm_prov(pdrb) if not pdrb.empty else pd.DataFrame()
    miskin_c = norm_prov(penduduk_miskin) if not penduduk_miskin.empty else pd.DataFrame()

    # --- Merge panel ---
    panel = pen_c.merge(ump_c, on=["Provinsi", "Tahun"], how="left")

    # TPT: gabungkan dari BPS TPT-TPAK (lebih detail) & fallback ke Semester
    if not tpt_bps_c.empty:
        panel = panel.merge(tpt_bps_c[["Provinsi", "Tahun", "TPT_BPS", "TPAK_BPS"]],
                            on=["Provinsi", "Tahun"], how="left")
    if not tpt_sem_c.empty:
        panel = panel.merge(tpt_sem_c[["Provinsi", "Tahun", "TPT"]],
                            on=["Provinsi", "Tahun"], how="left")

    # Buat kolom TPT_Final: gunakan TPT_BPS jika tersedia, fallback ke TPT semester
    if "TPT_BPS" in panel.columns and "TPT" in panel.columns:
        panel["TPT_Final"] = panel["TPT_BPS"].fillna(panel["TPT"])
        panel.drop(columns=["TPT_BPS", "TPT"], inplace=True)
        panel.rename(columns={"TPT_Final": "TPT"}, inplace=True)
    elif "TPT_BPS" in panel.columns:
        panel.rename(columns={"TPT_BPS": "TPT"}, inplace=True)

    # TPAK
    if "TPAK_BPS" in panel.columns:
        panel.rename(columns={"TPAK_BPS": "TPAK"}, inplace=True)

    # PDRB
    if not pdrb_c.empty:
        panel = panel.merge(pdrb_c[["Provinsi", "Tahun",
                                    "PDRB_HargaBerlaku", "PDRB_HargaKonstan"]],
                            on=["Provinsi", "Tahun"], how="left")

    # Persentase Penduduk Miskin
    if not miskin_c.empty:
        panel = panel.merge(miskin_c[["Provinsi", "Tahun", "Pct_Penduduk_Miskin"]],
                            on=["Provinsi", "Tahun"], how="left")

    # Inflasi rata-rata tahunan
    panel = panel.merge(inflasi_tahunan, on="Tahun", how="left")

    # --- Filter tahun yang memiliki data inti lengkap ---
    # Overlap: Pengeluaran (2017–2025) + UMP (2021–2025) = 2021–2025
    panel = panel[panel["Tahun"].between(2021, 2025)]

    # --- Drop baris dengan kolom kunci kosong ---
    key_cols = ["Total_Pengeluaran", "UMP", "Inflasi_Rata_Tahunan"]
    panel = panel.dropna(subset=key_cols)

    # --- Urutan kolom final ---
    col_order = [
        "Provinsi", "Tahun",
        "Pengeluaran_Makanan", "Pengeluaran_Bukan_Makanan", "Total_Pengeluaran",
        "UMP", "TPT", "TPAK",
        "PDRB_HargaBerlaku", "PDRB_HargaKonstan",
        "Pct_Penduduk_Miskin",
        "Inflasi_Rata_Tahunan",
    ]
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order].sort_values(["Tahun", "Provinsi"]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "clean_daya_beli.csv")
    panel.to_csv(out_path, index=False)

    print(f"   ✓ {len(panel)} baris × {len(panel.columns)} kolom")
    print(f"   ✓ Provinsi: {panel['Provinsi'].nunique()}, "
          f"Tahun: {sorted(panel['Tahun'].unique())}")
    print(f"   ✓ Kolom: {list(panel.columns)}")
    print(f"   ✓ Disimpan → {out_path}")
    return panel


# ===========================================================================
# SUMMARY HELPER
# ===========================================================================

def print_summary(df: pd.DataFrame, name: str):
    print(f"\n{'─'*65}")
    print(f"  Preview: {name}")
    print(f"{'─'*65}")
    print(f"  Shape   : {df.shape}")
    print(f"  Kolom   : {list(df.columns)}")
    null_dict = df.isnull().sum().to_dict()
    null_str = {k: v for k, v in null_dict.items() if v > 0}
    print(f"  Null (non-zero): {null_str if null_str else 'tidak ada'}")
    print(f"\n  5 baris pertama:")
    print(df.head().to_string(index=False))


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print("  PREPROCESSING PIPELINE v2 – Kelompok E ML UNAIR")
    print("=" * 65)
    print("\n>> Memuat semua dataset raw...\n")

    # Load semua dataset
    inflasi       = load_inflasi_mom()        # [1]
    ihk           = load_ihk()                # [2]
    # [3] Inflasi Y-to-D: referensi saja, tidak dimasukkan ke model
    bi_rate       = load_bi_rate()            # [4]
    ump           = load_ump()                # [5]
    pengeluaran   = load_pengeluaran()        # [6]
    usd_idr       = load_usd_idr()            # [7]
    pengangguran_sem = load_pengangguran_semester()  # [8]
    tpt_tpak      = load_tpt_tpak()           # [9]
    pdrb          = load_pdrb()               # [10]
    penduduk_miskin = load_penduduk_miskin()  # [11]
    inflasi_komp  = load_inflasi_komponen()   # [12]
    harga_minyak  = load_harga_minyak()       # [13]

    # Build output files
    ts    = build_inflasi_ts(inflasi, ihk, bi_rate, usd_idr,
                              inflasi_komp, harga_minyak)
    panel = build_daya_beli_panel(inflasi, ump, pengeluaran,
                                   pengangguran_sem, tpt_tpak,
                                   pdrb, penduduk_miskin)

    # Ringkasan
    print_summary(ts, "clean_inflasi_ts.csv")
    print_summary(panel, "clean_daya_beli.csv")

    print(f"\n{'='*65}")
    print("  ✅ Preprocessing selesai!")
    print(f"  Output disimpan di: datasets/processed/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
