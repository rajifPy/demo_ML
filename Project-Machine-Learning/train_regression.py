import os
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from data_pipeline import get_regression_pipeline_data

# 1. Ambil data anti-leakage dari pipeline
X_train, X_test, y_train_log, y_test_log, df = get_regression_pipeline_data(target_col="Total_Pengeluaran")

# Simpan nama kolom terlebih dahulu sebelum diubah menjadi NumPy Array
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]

# =========================================================
# KODE PERBAIKAN 1: Imputation dilakukan SEBELUM fitting model
# =========================================================
# Inisialisasi imputer dengan strategi rata-rata (mean)
imputer = SimpleImputer(strategy='mean')

# Fit ke X_train dan transform nilainya
X_train = imputer.fit_transform(X_train)

# Jangan lupa transform juga untuk X_test agar sinkron saat prediksi
X_test = imputer.transform(X_test)
# =========================================================

# 2. Inisialisasi dan Latih Model Ridge Regression
# Menggunakan Alpha=1.0 untuk regularisasi L2 guna mencegah overfitting data ekonomi
model_ridge = Ridge(alpha=1.0)
print("\n[INFO] Memulai pelatihan model Ridge Regression...")
model_ridge.fit(X_train, y_train_log)

# 3. Evaluasi Model pada Data Test
y_pred_log = model_ridge.predict(X_test)
mse = mean_squared_error(y_test_log, y_pred_log)
r2 = r2_score(y_test_log, y_pred_log)

print("\n" + "="*40)
print(" EVALUASI MODEL REGRESI (DAYA BELI)")
print("="*40)
print(f" ✓ Mean Squared Error (MSE) : {mse:.6f}")
print(f" ✓ R-squared (R²) Score     : {r2:.4f} ({r2*100:.2f}%)")
print("-" * 40)

# KODE PERBAIKAN 2: Menampilkan koefisien menggunakan nama kolom yang sudah disimpan
print("\n[KONTRIBUSI FITUR TERHADAP DAYA BELI]:")
for col, coef in zip(feature_names, model_ridge.coef_):
    print(f" ↳ {col:<25} : {coef:.4f}")

# 4. Simpan Model langsung ke folder proyek Django
export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(export_dir, exist_ok=True)

# Simpan model dengan nama yang dicari oleh Django views.py
ridge_save_path = os.path.join(export_dir, "best_daya_beli_ridge.pkl")
with open(ridge_save_path, "wb") as f:
    pickle.dump(model_ridge, f)

# Simpan juga imputernya di folder yang sama agar sinkron saat handling NaN di web
imputer_save_path = os.path.join(export_dir, "imputer_regression.pkl")
with open(imputer_save_path, "wb") as f:
    pickle.dump(imputer, f)

print(f"\n✓ Model Ridge (5 Fitur) sukses diekspor langsung ke: {ridge_save_path}")
print("✓ Dashboard Django otomatis menggunakan model terbaru ini tanpa perlu ubah codingan web!")