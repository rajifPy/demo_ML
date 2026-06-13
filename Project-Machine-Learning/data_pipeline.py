"""
=============================================================================
  DATA PIPELINE - FINAL OPTIMIZED
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets", "processed")


def get_forecasting_pipeline_data(seq_length=12, lag_features=True):
    """
    Pipeline untuk Model 1 (Forecasting Inflasi).
    Menggunakan semua fitur komoditas global dan makro ekonomi yang kaya.
    """
    print("\n" + "="*50)
    print("  Forecasting Data Pipeline (Time Series)")
    print("="*50)
    
    path = os.path.join(OUT_DIR, "clean_inflasi_ts.csv")
    if not os.path.exists(path):
        # Fallback jika dijalankan di lokal folder yang sama
        path = "clean_inflasi_ts.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} tidak ditemukan.")
        
    df = pd.read_csv(path)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)
    
    # 1. Imputasi yang aman untuk data deret waktu keuangan/ekonomi
    # Menggunakan metode interpolasi linear agar tidak merusak tren data historis
    df = df.interpolate(method='linear').ffill().bfill()
    
    # Target adalah Inflasi_MoM
    target_col = "Inflasi_MoM"
    
    # Mengambil semua kolom numerik kecuali Tanggal, Bulan, Tahun, dan Variasi Inflasi lainnya untuk mencegah leakage
    exclude_cols = ["Tanggal", "Bulan", "Tahun", "Inflasi_YoY", "Inflasi_YtD", 
                    "Inflasi_Umum_MoM", "Inflasi_Inti_MoM", "Inflasi_HargaDiatur_MoM", "Inflasi_Bergejolak_MoM"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
    
    # Letakkan target di index pertama untuk mempermudah slicing
    all_cols = [target_col] + feature_cols
    raw_data = df[all_cols].values
    
    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(raw_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # Fit Scaler HANYA di data Train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(raw_data[:train_end])
    
    all_scaled = scaler.transform(raw_data)
    
    # 3. Pembuatan Sequence (Bisa untuk LSTM atau diekstrak jadi tabular untuk LightGBM/XGBoost)
    def create_sequences(data, start_idx, end_idx, seq_len):
        xs, ys = [], []
        for i in range(start_idx, end_idx - seq_len):
            x = data[i:(i + seq_len)]
            y = data[i + seq_len, 0] # Index 0 adalah Target (Inflasi_MoM)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X_train, y_train = create_sequences(all_scaled, 0, train_end, seq_length)
    X_val, y_val = create_sequences(all_scaled, train_end - seq_length, val_end, seq_length)
    X_test, y_test = create_sequences(all_scaled, val_end - seq_length, n, seq_length)
    
    print(f"   ✓ Fitur yang digunakan: {len(feature_cols)} Indikator Ekonomi & Komoditas")
    print(f"   ✓ Rentang Train: {df['Tanggal'].iloc[0].date()} s.d {df['Tanggal'].iloc[train_end-1].date()}")
    print(f"   ✓ X_train : {X_train.shape}, y_train: {y_train.shape}")
    print(f"   ✓ X_test  : {X_test.shape}, y_test : {y_test.shape}")
    print("-" * 50)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df


def get_regression_pipeline_data(target_col="Total_Pengeluaran"):
    """
    Pipeline untuk Model 2 (Regresi Dampak Daya Beli).
    Menangani data panel provinsi dan mengonversi kolom kategorikal dengan aman.
    """
    print("\n" + "="*50)
    print(f"  Regression Pipeline (Target: {target_col})")
    print("="*50)
    
    path = os.path.join(OUT_DIR, "clean_daya_beli.csv")
    if not os.path.exists(path):
        path = "clean_daya_beli.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} tidak ditemukan.")
        
    df = pd.read_csv(path)
    
    # Pastikan target valid
    if target_col not in df.columns:
        raise ValueError(f"Target kolom {target_col} tidak ditemukan di dataset.")
        
    y = df[target_col]
    
    # Fitur Utama Analisis Dampak
    core_features = ["UMP", "TPT", "Inflasi_Rata_Tahunan", "PDRB_HargaKonstan", "Pct_Penduduk_Miskin"]
    X = df[core_features].copy()
    
    # Jika ingin menyertakan efek wilayah (Provinsi), kita lakukan One-Hot Encoding
    # Untuk menghindari leakage pada tingkat kemiskinan/UMP, kita split dulu baru imputasi
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    # Imputasi Mean TPT berbasis data Train saja
    mean_tpt = X_train["TPT"].mean()
    X_train["TPT"] = X_train["TPT"].fillna(mean_tpt)
    X_test["TPT"] = X_test["TPT"].fillna(mean_tpt)
    
    # Log-Transform untuk menormalkan distribusi data ekonomi berskala besar (UMP & PDRB)
    for col in ["UMP", "PDRB_HargaKonstan"]:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])
        
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    print(f"   ✓ Total observasi panel data: {len(df)}")
    print(f"   ✓ X_train: {X_train.shape}, y_train: {y_train_log.shape}")
    print(f"   ✓ X_test : {X_test.shape},  y_test : {y_test_log.shape}")
    print("-" * 50)
    
    return X_train, X_test, y_train_log, y_test_log, df


if __name__ == "__main__":
    print("\nMenyelesaikan uji coba (Dry Run) Pipeline...")
    try:
        lstm_data = get_forecasting_pipeline_data(seq_length=12)
        print("✓ Forecasting Pipeline OK.")
    except Exception as e:
        print(f"✗ Gagal Forecasting Pipeline: {e}")
        
    try:    
        reg_data = get_regression_pipeline_data(target_col="Total_Pengeluaran")
        print("✓ Regression Pipeline OK.\n")
    except Exception as e:
        print(f"✗ Gagal Regression Pipeline: {e}")