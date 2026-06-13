"""
=============================================================================
  EKSPLORASI & VISUALISASI DATASET
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
  Kelompok E – Machine Learning SD-A1, Universitas Airlangga
=============================================================================

Dataset yang digunakan (8 dataset):
  1. Indeks Harga Konsumen / IHK     (BPS) – per kota, tahunan CSV
  2. Inflasi Bulanan M-to-M          (BPS) – per kota, tahunan CSV
  3. BI Rate / Data Inflasi          (Bank Indonesia) – bulanan Excel
  4. Kurs USD/IDR Historis           (Investing.com) – harian CSV
  5. Upah Minimum Provinsi / UMP     (BPS) – tahunan CSV
  6. Rata-rata Pengeluaran per Kapita (BPS) – tahunan CSV
  7. Tingkat Pengangguran Terbuka    (Open Data Jabar) – semesteran CSV
  (Dataset 8 = Inflasi Y-to-D, digunakan sebagai referensi saja)

Anggota Kelompok:
  - Muhammad Rajif Al Farikhi    (162112133008)
  - Sahrul Adicandra Effendy     (164231013)
  - Semaya David Petroes Putra   (164231048)
  - Adrina Firda Marwah          (164231087)
  - Okan Athallah Maredith       (164231088)
=============================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Kamus bulan Indonesia → angka
# ---------------------------------------------------------------------------
BULAN_MAP = {
    "Januari": 1,  "Februari": 2,  "Maret": 3,    "April": 4,
    "Mei": 5,      "Juni": 6,      "Juli": 7,     "Agustus": 8,
    "September": 9,"Oktober": 10,  "November": 11,"Desember": 12,
}


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _parse_indo_date(date_str: str) -> pd.Timestamp:
    try:
        parts = str(date_str).strip().split()
        if len(parts) == 2:
            bulan = BULAN_MAP.get(parts[0], None)
            if bulan:
                return pd.Timestamp(year=int(parts[1]), month=bulan, day=1)
    except Exception:
        pass
    return pd.NaT


def _to_float_id(val) -> float:
    """Konversi angka format Indonesia (1.234,56 → 1234.56)."""
    try:
        s = str(val).strip().replace("%", "").replace(" ", "")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "." in s and s.count(".") > 1:
            s = s.replace(".", "")
        return float(s)
    except Exception:
        return np.nan


def _extract_year_from_filename(filename: str):
    try:
        stem = os.path.splitext(os.path.basename(filename))[0]
        last_part = stem.split(",")[-1].strip()
        if last_part.isdigit() and len(last_part) == 4:
            return int(last_part)
    except Exception:
        pass
    return None


def _find_indonesia_row(df: pd.DataFrame, col_idx: int = 0):
    col = df.iloc[:, col_idx].astype(str).str.upper().str.strip()
    match = df[col == "INDONESIA"]
    return match.iloc[0] if not match.empty else None


# ===========================================================================
# DATASET LOADERS
# ===========================================================================

def load_ikk(base_path: str) -> pd.DataFrame:
    """Dataset 1 – Indeks Harga Konsumen (IHK) Nasional (2005–2019)"""
    print("1. Memproses Indeks Harga Konsumen (IHK)...")
    files = glob.glob(os.path.join(base_path, "Indeks Harga Konsumen (Umum)", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia_row(df, 0)
            if row is None:
                continue
            for bulan_nama, bulan_angka in BULAN_MAP.items():
                if bulan_nama in df.columns:
                    val = _to_float_id(row[bulan_nama])
                    if not np.isnan(val):
                        records.append({
                            "Tanggal": pd.Timestamp(year=tahun, month=bulan_angka, day=1),
                            "IHK": val,
                        })
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    df_ikk = pd.DataFrame(records).sort_values("Tanggal").drop_duplicates("Tanggal").set_index("Tanggal")
    print(f"   ✓ {len(df_ikk)} observasi ({df_ikk.index.min().year if not df_ikk.empty else '?'}–{df_ikk.index.max().year if not df_ikk.empty else '?'})")
    return df_ikk


def load_inflasi_mom(base_path: str) -> pd.DataFrame:
    """Dataset 2 – Inflasi Bulanan M-to-M (2005–2026)"""
    print("\n2. Memproses Inflasi Bulanan (M-to-M)...")
    files = glob.glob(os.path.join(base_path, "Inflasi Bulanan", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia_row(df, 0)
            if row is None:
                continue
            for bulan_nama, bulan_angka in BULAN_MAP.items():
                if bulan_nama in df.columns:
                    val = _to_float_id(row[bulan_nama])
                    if not np.isnan(val):
                        records.append({
                            "Tanggal": pd.Timestamp(year=tahun, month=bulan_angka, day=1),
                            "Inflasi_MoM": val,
                        })
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    df_inflasi = (
        pd.DataFrame(records)
        .sort_values("Tanggal")
        .drop_duplicates("Tanggal")
        .set_index("Tanggal")
    )
    print(f"   ✓ {len(df_inflasi)} observasi ({df_inflasi.index.min().year if not df_inflasi.empty else '?'}–{df_inflasi.index.max().year if not df_inflasi.empty else '?'})")
    return df_inflasi


def load_bi_rate(base_path: str) -> pd.DataFrame:
    """Dataset 3 – BI Rate / Data Inflasi Bank Indonesia"""
    print("\n3. Memproses BI Rate (Data Inflasi BI)...")
    bi_path = os.path.join(base_path, "BI Rate (Data Inflasi)", "Data Inflasi.xlsx")
    try:
        # Header sebenarnya ada di baris ke-3 (0-indexed), skiprows=3 lalu header=0
        df = pd.read_excel(bi_path, skiprows=3, header=0)
        # Jika header terbaca sebagai data (kolom unnamed), header ada di baris data pertama
        if "Periode" not in df.columns:
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        df = df[["Periode", "Data Inflasi"]].dropna()
        df["Data Inflasi"] = df["Data Inflasi"].apply(_to_float_id)
        df["Tanggal"] = df["Periode"].apply(_parse_indo_date)
        df = df.dropna(subset=["Tanggal", "Data Inflasi"]).sort_values("Tanggal").set_index("Tanggal")
        print(f"   ✓ {len(df)} observasi ({df.index.min().year}–{df.index.max().year})")
        return df[["Data Inflasi"]]
    except Exception as e:
        print(f"   ⚠ Gagal memuat BI Rate: {e}")
        return pd.DataFrame()


def load_usd_idr(base_path: str) -> pd.DataFrame:
    """Dataset 4 – Kurs USD/IDR Historis (harian → resample bulanan)"""
    print("\n4. Memproses Kurs USD/IDR...")
    path = os.path.join(base_path, "Data Historis USD_IDR", "Data Historis USD_IDR.csv")
    try:
        df = pd.read_csv(path, dtype=str)
        df.rename(columns={df.columns[0]: "Tanggal", df.columns[1]: "Kurs"}, inplace=True)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
        df["Kurs"] = df["Kurs"].apply(_to_float_id)
        df = df.dropna(subset=["Tanggal", "Kurs"]).set_index("Tanggal").sort_index()
        df_monthly = df[["Kurs"]].resample("MS").mean()
        print(f"   ✓ {len(df_monthly)} bulan setelah resample ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"   ⚠ Gagal memuat USD/IDR: {e}")
        return pd.DataFrame()


def load_ump(base_path: str) -> pd.DataFrame:
    """Dataset 5 – Upah Minimum Provinsi (UMP) (2021–2025)"""
    print("\n5. Memproses Upah Minimum Provinsi (UMP)...")
    files = glob.glob(os.path.join(base_path, "Upah Minimum Provinsi", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=2, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi", df.columns[1]: "UMP"}, inplace=True)
            df = df.dropna(subset=["Provinsi", "UMP"])
            df = df[~df["Provinsi"].str.strip().str.upper().isin(["PROVINSI", "INDONESIA"])]
            df["Provinsi"] = df["Provinsi"].str.strip().str.title()
            df["UMP"] = df["UMP"].apply(_to_float_id)
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "UMP", "Tahun"]].dropna())
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    ump_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    n_tahun = ump_df["Tahun"].nunique() if not ump_df.empty else 0
    print(f"   ✓ {len(ump_df)} baris dari {n_tahun} tahun ({', '.join(map(str, sorted(ump_df['Tahun'].unique()))) if not ump_df.empty else '-'})")
    return ump_df


def load_pengeluaran(base_path: str) -> pd.DataFrame:
    """Dataset 6 – Rata-rata Pengeluaran per Kapita Sebulan (2017–2025)"""
    print("\n6. Memproses Rata-rata Pengeluaran per Kapita...")
    folder = "Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan"
    files = glob.glob(os.path.join(base_path, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            nasional = df[df["Provinsi"].astype(str).str.strip().str.upper() == "INDONESIA"]
            if not nasional.empty:
                total_col = df.columns[-1]
                val = _to_float_id(nasional[total_col].values[0])
                if not np.isnan(val):
                    records.append({"Tahun": tahun, "Total_Pengeluaran": val})
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    pengeluaran_df = pd.DataFrame(records).sort_values("Tahun").set_index("Tahun")
    print(f"   ✓ {len(pengeluaran_df)} tahun ({', '.join(map(str, pengeluaran_df.index.tolist()))})")
    return pengeluaran_df


def load_pengangguran(base_path: str) -> pd.DataFrame:
    """Dataset 7 – Tingkat Pengangguran Terbuka per Provinsi per Tahun (2020–2025)"""
    print("\n7. Memproses Tingkat Pengangguran Terbuka (TPT)...")
    path = os.path.join(
        base_path,
        "Tingkat Pengangguran Terbuka Berdasarkan Semester dan Provinsi di Indonesia",
        "disnakertrans-od_21012_tingkat_pengangguran_terbuka_brdsrkn_semester_prov_v1_data.csv"
    )
    try:
        df = pd.read_csv(path, dtype=str)
        df["tingkat_pengangguran_terbuka"] = df["tingkat_pengangguran_terbuka"].apply(_to_float_id)
        df["tahun"] = df["tahun"].astype(int)
        # Rata-rata nasional per tahun (avg Feb + Agustus semua provinsi)
        nasional = (df.groupby("tahun")["tingkat_pengangguran_terbuka"]
                    .mean()
                    .reset_index()
                    .rename(columns={"tahun": "Tahun", "tingkat_pengangguran_terbuka": "TPT"})
                    .set_index("Tahun"))
        print(f"   ✓ {len(nasional)} tahun ({nasional.index.min()}–{nasional.index.max()})")
        return nasional
    except Exception as e:
        print(f"   ⚠ Gagal memuat TPT: {e}")
        return pd.DataFrame()


# ===========================================================================
# RINGKASAN DATASET
# ===========================================================================

def print_summary(datasets: dict):
    print("\n" + "=" * 70)
    print("  RINGKASAN DATASET")
    print("=" * 70)
    descs = {
        "ikk":           ("Indeks Harga Konsumen (IHK)",               "BPS",            "Bulanan / per Kota",   "IHK (angka indeks)"),
        "inflasi_mom":   ("Inflasi Bulanan M-to-M",                    "BPS",            "Bulanan / per Kota",   "Inflasi MoM (%)"),
        "bi_rate":       ("BI Rate / Data Inflasi Bank Indonesia",      "Bank Indonesia", "Bulanan",              "Inflasi YoY (%)"),
        "usd_idr":       ("Kurs USD/IDR Historis",                     "Investing.com",  "Bulanan (resample)",   "Kurs (Rp/USD)"),
        "ump":           ("Upah Minimum Provinsi (UMP)",                "BPS",            "Tahunan / Provinsi",   "UMP (Rp/bulan)"),
        "pengeluaran":   ("Rata-rata Pengeluaran per Kapita Sebulan",   "BPS",            "Tahunan / Nasional",   "Rupiah/kapita/bulan"),
        "pengangguran":  ("Tingkat Pengangguran Terbuka (TPT)",         "Open Data Jabar","Semesteran / Provinsi","% angkatan kerja"),
    }
    for key, df in datasets.items():
        name, sumber, frekuensi, satuan = descs.get(key, (key, "-", "-", "-"))
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.index, pd.DatetimeIndex):
                rentang = f"{df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')}"
            elif df.index.name in ["Tahun"]:
                rentang = f"{df.index.min()} – {df.index.max()}"
            else:
                rentang = f"{df['Tahun'].min()} – {df['Tahun'].max()}" if "Tahun" in df.columns else "-"
            rows = len(df)
        else:
            rentang, rows = "-", 0
        print(f"\n  📦 {name}")
        print(f"     Sumber   : {sumber}")
        print(f"     Frekuensi: {frekuensi}")
        print(f"     Satuan   : {satuan}")
        print(f"     Rentang  : {rentang}")
        print(f"     Jumlah   : {rows:,} baris")
    print("\n" + "=" * 70)


# ===========================================================================
# VISUALIZATION — 8 panel (4 × 2), dark theme
# ===========================================================================

C = {
    "inflasi":     "#E63946",
    "ihk":         "#4361EE",
    "bi_rate":     "#7B2FBE",
    "usd_idr":     "#F4A261",
    "ump_avg":     "#2DC653",
    "ump_bar":     "#06D6A0",
    "pengeluaran": "#118AB2",
    "tpt":         "#FFB703",
    "bg":          "#0D1117",
    "panel":       "#161B22",
    "text":        "#E6EDF3",
    "subtext":     "#8B949E",
    "grid":        "#EEEEEE",
}


def _style_ax(ax, title: str, ylabel: str = "", xlabel: str = ""):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, color=C["text"], fontsize=10.5, fontweight="bold", pad=7)
    ax.set_ylabel(ylabel, color=C["subtext"], fontsize=8)
    ax.set_xlabel(xlabel, color=C["subtext"], fontsize=8)
    ax.tick_params(colors=C["subtext"], labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["panel"])
    ax.grid(True, color=C["grid"], alpha=0.12, linestyle="--", linewidth=0.6)


def create_visualizations(datasets: dict, output_dir: str = "datasets"):
    print("\n=== MEMBUAT VISUALISASI DASHBOARD (8 Panel) ===")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": C["bg"],
        "text.color": C["text"],
    })

    fig = plt.figure(figsize=(22, 16), facecolor=C["bg"])
    fig.suptitle(
        "Dashboard Analisis Dataset: Inflasi & Daya Beli Indonesia",
        fontsize=19, fontweight="bold", color=C["text"], y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "Kelompok E – Machine Learning SD-A1, Universitas Airlangga  |  Sumber: BPS, Bank Indonesia, Investing.com, Open Data Jabar",
        ha="center", fontsize=8.5, color=C["subtext"],
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.50, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ── Panel 1 (baris 0, kol 0): Inflasi MoM ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    inflasi = datasets.get("inflasi_mom")
    if inflasi is not None and not inflasi.empty:
        ax1.fill_between(inflasi.index, inflasi["Inflasi_MoM"], alpha=0.22, color=C["inflasi"])
        ax1.plot(inflasi.index, inflasi["Inflasi_MoM"],
                 color=C["inflasi"], linewidth=1.4, label="Inflasi MoM (%)")
        ax1.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.45)
        mean_val = inflasi["Inflasi_MoM"].mean()
        ax1.axhline(mean_val, color=C["inflasi"], linewidth=0.8,
                    linestyle=":", alpha=0.7, label=f"Rata-rata: {mean_val:.2f}%")
        ax1.legend(fontsize=7.5, labelcolor=C["text"], loc="upper right")
    _style_ax(ax1, "① Inflasi Bulanan Nasional (M-to-M)", ylabel="Inflasi (%)")

    # ── Panel 2 (baris 0, kol 1): IHK Nasional ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ikk = datasets.get("ikk")
    if ikk is not None and not ikk.empty:
        ax2.plot(ikk.index, ikk["IHK"],
                 color=C["ihk"], linewidth=2, label="IHK Nasional")
        ax2.fill_between(ikk.index, ikk["IHK"].min(), ikk["IHK"],
                         alpha=0.15, color=C["ihk"])
        ax2.legend(fontsize=7.5, labelcolor=C["text"])
    _style_ax(ax2, "② Indeks Harga Konsumen (IHK) Nasional (2005–2019)", ylabel="Angka Indeks")

    # ── Panel 3 (baris 1, kol 0): BI Rate ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bi_rate = datasets.get("bi_rate")
    if bi_rate is not None and not bi_rate.empty:
        ax3.plot(bi_rate.index, bi_rate["Data Inflasi"],
                 color=C["bi_rate"], linewidth=2, label="Data Inflasi BI (%)")
        ax3.fill_between(bi_rate.index, bi_rate["Data Inflasi"],
                         alpha=0.18, color=C["bi_rate"])
        ax3.legend(fontsize=7.5, labelcolor=C["text"])
    _style_ax(ax3, "③ Data Inflasi Bank Indonesia (BI Rate)", ylabel="Inflasi YoY (%)")

    # ── Panel 4 (baris 1, kol 1): Kurs USD/IDR ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    usd_idr = datasets.get("usd_idr")
    if usd_idr is not None and not usd_idr.empty:
        ax4.plot(usd_idr.index, usd_idr["Kurs"],
                 color=C["usd_idr"], linewidth=1.6, label="USD/IDR (rata-rata bulanan)")
        ax4.fill_between(usd_idr.index, usd_idr["Kurs"].min(), usd_idr["Kurs"],
                         alpha=0.18, color=C["usd_idr"])
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax4.legend(fontsize=7.5, labelcolor=C["text"])
    _style_ax(ax4, "④ Kurs USD/IDR Historis (Bulanan)", ylabel="Rp / USD")

    # ── Panel 5 (baris 2, kol 0): UMP rata-rata nasional per tahun ────────
    ax5 = fig.add_subplot(gs[2, 0])
    ump = datasets.get("ump")
    if ump is not None and not ump.empty:
        ump_avg = ump.groupby("Tahun")["UMP"].mean() / 1_000_000
        bars = ax5.bar(ump_avg.index.astype(str), ump_avg.values,
                       color=C["ump_avg"], alpha=0.85, width=0.5)
        for bar in bars:
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{bar.get_height():.2f}",
                     ha="center", va="bottom", fontsize=7.5, color=C["subtext"])
    _style_ax(ax5, "⑤ Rata-Rata UMP Nasional per Tahun", ylabel="Juta Rupiah / Bulan")

    # ── Panel 6 (baris 2, kol 1): UMP Top 10 Provinsi ────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    if ump is not None and not ump.empty:
        latest_year = int(ump["Tahun"].max())
        top10 = (ump[ump["Tahun"] == latest_year]
                 .sort_values("UMP", ascending=True)
                 .tail(10))
        colors_bar = sns.color_palette("YlOrRd", len(top10))
        ax6.barh(top10["Provinsi"], top10["UMP"] / 1_000_000, color=colors_bar, alpha=0.9)
        ax6.set_xlabel("Juta Rupiah / Bulan", color=C["subtext"], fontsize=8)
    _style_ax(ax6, f"⑥ 10 Provinsi UMP Tertinggi ({latest_year if ump is not None and not ump.empty else '-'})")

    # ── Panel 7 (baris 3, kol 0): Pengeluaran per Kapita Nasional ─────────
    ax7 = fig.add_subplot(gs[3, 0])
    pengeluaran = datasets.get("pengeluaran")
    if pengeluaran is not None and not pengeluaran.empty:
        colors_p = sns.color_palette("Blues_d", len(pengeluaran))
        bars = ax7.bar(
            pengeluaran.index.astype(str),
            pengeluaran["Total_Pengeluaran"] / 1_000,
            color=colors_p, alpha=0.9,
        )
        ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax7.tick_params(axis="x", rotation=20)
        for bar in bars:
            ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                     f"{bar.get_height():,.0f}",
                     ha="center", va="bottom", fontsize=6.5, color=C["subtext"])
    _style_ax(ax7, "⑦ Rata-rata Pengeluaran per Kapita Nasional", ylabel="Ribu Rp / Kapita / Bulan")

    # ── Panel 8 (baris 3, kol 1): TPT Nasional ────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    tpt = datasets.get("pengangguran")
    if tpt is not None and not tpt.empty:
        ax8.plot(tpt.index.astype(str), tpt["TPT"],
                 color=C["tpt"], linewidth=2.2, marker="o", markersize=6, label="TPT Nasional (%)")
        ax8.fill_between(tpt.index.astype(str), tpt["TPT"],
                         alpha=0.2, color=C["tpt"])
        for x, y in zip(tpt.index.astype(str), tpt["TPT"]):
            ax8.text(x, y + 0.05, f"{y:.2f}%",
                     ha="center", fontsize=7.5, color=C["tpt"])
        ax8.legend(fontsize=7.5, labelcolor=C["text"])
    _style_ax(ax8, "⑧ Tingkat Pengangguran Terbuka (TPT) Nasional", ylabel="% Angkatan Kerja")

    # ── Simpan ─────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "visualisasi_dataset.png")
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"✓ Visualisasi disimpan → {out_path}")
    return out_path


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    datasets_dir = "datasets"

    print("=" * 70)
    print("  PROYEK: PREDIKSI INFLASI & DAMPAKNYA TERHADAP DAYA BELI")
    print("  Kelompok E – Machine Learning SD-A1, Universitas Airlangga")
    print("=" * 70 + "\n")
    print("=== MEMUAT SEMUA DATASET ===\n")

    datasets = {
        "ikk":          load_ikk(datasets_dir),
        "inflasi_mom":  load_inflasi_mom(datasets_dir),
        "bi_rate":      load_bi_rate(datasets_dir),
        "usd_idr":      load_usd_idr(datasets_dir),
        "ump":          load_ump(datasets_dir),
        "pengeluaran":  load_pengeluaran(datasets_dir),
        "pengangguran": load_pengangguran(datasets_dir),
    }

    print_summary(datasets)
    create_visualizations(datasets, datasets_dir)

    print("\n✅ Eksplorasi dataset selesai!")


if __name__ == "__main__":
    main()