import numpy as np
import pandas as pd


NOMINAL_TARGET_COLUMN = "Total_Pengeluaran"
TARGET_COLUMN = "Total_Pengeluaran_Riil"
TARGET_LABEL = "Pengeluaran riil per kapita per bulan"
PROVINCE_COLUMN = "Provinsi"
YEAR_COLUMN = "Tahun"

RAW_COMPONENT_COLUMNS = [
    "Pengeluaran_Makanan",
    "Pengeluaran_Bukan_Makanan",
]

BASE_NUMERIC_FEATURES = [
    "UMP",
    "TPT",
    "TPAK",
    "PDRB_HargaBerlaku",
    "PDRB_HargaKonstan",
    "Pct_Penduduk_Miskin",
    "Inflasi_Rata_Tahunan",
    "Gini_Rasio",
    "IPM",
    "Garis_Kemiskinan",
    "Jumlah_Penduduk",
    "Pct_Populasi",
    "Pct_Akses_Air_Bersih",
    "Protein_gram_per_hari",
    "Inflasi_WB_Annual",
    "GDP_PerCapita_PPP",
    "Pct_Unemployment_WB",
    "Poverty_Headcount_Pct",
]

DERIVED_NUMERIC_FEATURES = [
    "Year_Index",
    "Inflation_Deflator",
    "Real_UMP",
    "Real_UMP_Growth",
    "PDRB_HargaKonstan_Growth",
    "TPT_Growth",
    "UMP_x_PDRB",
    "Inflasi_x_TPT",
    "Log_PDRB",
    "Log_UMP",
    "Prev_Total_Pengeluaran_Riil",
]

MODEL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
DEPLOYMENT_NUMERIC_FEATURE_CANDIDATES = [
    "TPT",
    "TPAK",
    "PDRB_HargaKonstan",
    "Pct_Penduduk_Miskin",
    "Inflasi_Rata_Tahunan",
    "Gini_Rasio",
    "IPM",
    "Garis_Kemiskinan",
    "Jumlah_Penduduk",
    "Pct_Populasi",
    "Pct_Akses_Air_Bersih",
    "Protein_gram_per_hari",
    "Inflasi_WB_Annual",
    "GDP_PerCapita_PPP",
    "Pct_Unemployment_WB",
    "Poverty_Headcount_Pct",
    "Year_Index",
    "Inflation_Deflator",
    "Real_UMP",
    "Real_UMP_Growth",
    "PDRB_HargaKonstan_Growth",
    "TPT_Growth",
    "UMP_x_PDRB",
    "Inflasi_x_TPT",
    "Log_PDRB",
    "Log_UMP",
    "Prev_Total_Pengeluaran_Riil",
]
MODEL_CATEGORICAL_FEATURES = [PROVINCE_COLUMN]

TRAIN_END_YEAR = 2024
TEST_START_YEAR = 2025


def prepare_daya_beli_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.sort_values([PROVINCE_COLUMN, YEAR_COLUMN]).reset_index(drop=True)
    prepared["Year_Index"] = prepared[YEAR_COLUMN] - int(prepared[YEAR_COLUMN].min())

    inflation_source = prepared["Inflasi_WB_Annual"].where(
        prepared["Inflasi_WB_Annual"].notna(),
        prepared["Inflasi_Rata_Tahunan"],
    )
    inflasi_decimal = inflation_source / 100.0
    prepared["Inflation_Deflator"] = 1.0 + inflasi_decimal
    prepared[TARGET_COLUMN] = prepared[NOMINAL_TARGET_COLUMN] / prepared["Inflation_Deflator"].clip(lower=1e-9)
    prepared["Real_UMP"] = prepared["UMP"] / (1 + inflasi_decimal)

    grouped = prepared.groupby(PROVINCE_COLUMN, group_keys=False)
    prepared["Prev_Total_Pengeluaran_Riil"] = grouped[TARGET_COLUMN].shift(1)
    prepared["Prev_UMP"] = grouped["UMP"].shift(1)
    prepared["Prev_Inflation_Deflator"] = grouped["Inflation_Deflator"].shift(1)
    prepared["Prev_Real_UMP"] = prepared["Prev_UMP"] / prepared["Prev_Inflation_Deflator"].clip(lower=1e-9)
    prepared["Prev_PDRB_HargaKonstan"] = grouped["PDRB_HargaKonstan"].shift(1)
    prepared["Prev_TPT"] = grouped["TPT"].shift(1)

    prepared["Real_UMP_Growth"] = grouped["Real_UMP"].pct_change(fill_method=None) * 100.0
    prepared["PDRB_HargaKonstan_Growth"] = grouped["PDRB_HargaKonstan"].pct_change(fill_method=None) * 100.0
    prepared["TPT_Growth"] = grouped["TPT"].pct_change(fill_method=None) * 100.0

    prepared["Real_UMP_Growth"] = prepared["Real_UMP_Growth"].fillna(0.0)
    prepared["PDRB_HargaKonstan_Growth"] = prepared["PDRB_HargaKonstan_Growth"].fillna(0.0)
    prepared["TPT_Growth"] = prepared["TPT_Growth"].fillna(0.0)
    prepared["Prev_Total_Pengeluaran_Riil"] = prepared["Prev_Total_Pengeluaran_Riil"].fillna(prepared[TARGET_COLUMN])

    prepared["UMP_x_PDRB"] = prepared["Real_UMP"] * prepared["PDRB_HargaKonstan"]
    prepared["Inflasi_x_TPT"] = prepared["Inflasi_Rata_Tahunan"] * prepared["TPT"]
    prepared["Log_PDRB"] = np.log1p(prepared["PDRB_HargaKonstan"].clip(lower=0.0))
    prepared["Log_UMP"] = np.log1p(prepared["Real_UMP"].clip(lower=0.0))

    return prepared


def get_available_numeric_features(df: pd.DataFrame) -> list[str]:
    return [feature for feature in MODEL_NUMERIC_FEATURES if feature in df.columns]


def get_available_deployment_features(df: pd.DataFrame) -> list[str]:
    return [feature for feature in DEPLOYMENT_NUMERIC_FEATURE_CANDIDATES if feature in df.columns]


def build_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    prepared = prepare_daya_beli_dataframe(df)
    numeric_features = get_available_deployment_features(prepared)
    required_columns = [TARGET_COLUMN, YEAR_COLUMN] + MODEL_CATEGORICAL_FEATURES + numeric_features
    model_df = prepared[required_columns].copy()
    model_df = model_df.dropna(subset=[TARGET_COLUMN, YEAR_COLUMN, PROVINCE_COLUMN])
    return model_df, numeric_features, MODEL_CATEGORICAL_FEATURES.copy()
