import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_pipeline import get_forecasting_pipeline_data

# 1. Ambil data anti-leakage dari pipeline
(X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df = get_forecasting_pipeline_data(seq_length=12)

# 2. Bangun Arsitektur Model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1) # Output berupa nilai prediksi kontinu (Inflasi MoM)
])

# 3. Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 4. Set Callback agar tidak Overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n[INFO] Memulai pelatihan model LSTM...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# 5. Evaluasi singkat pada data Test
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[HASIL] Evaluasi Data Test -> Loss (MSE): {test_loss:.6f}, MAE: {test_mae:.6f}")

# 6. Simpan Model dan Scaler ke dalam folder proyek
os.makedirs("saved_models", exist_ok=True)

# Simpan Arsitektur & Bobot LSTM
model.save("saved_models/lstm_inflation_model.h5")
# Simpan Scaler (Penting untuk proses inverse transform saat prediksi di dashboard web)
with open("saved_models/scaler_lstm.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✓ Model LSTM dan Scaler berhasil disimpan di folder 'saved_models/'!")