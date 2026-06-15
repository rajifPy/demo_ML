# EcoDash - Prediksi Inflasi dan Proksi Daya Beli Indonesia

EcoDash adalah dashboard analitik ekonomi Indonesia yang mengintegrasikan dua modul inti:

1. forecast inflasi multi-horizon untuk membaca arah tekanan harga
2. estimasi pengeluaran riil per kapita per provinsi sebagai proksi daya beli rumah tangga

Repositori ini diposisikan untuk demonstrasi profesional, analisis akademik, dan eksplorasi kebijakan berbasis data resmi.

## Ringkasan sistem

EcoDash saat ini memiliki empat area utama:

- `Home (/)`: landing page dengan headline forecast publik 1 bulan, kurs USD/IDR harian, dan ringkasan kapabilitas
- `Dashboard (/dashboard/)`: KPI operasional, ringkasan istilah inflasi, panel USD/IDR live, dan akses cepat ke modul utama
- `Forecasting (/forecasting/)`: forecast inflasi untuk horizon `1M`, `3M`, `6M`, dan `12M`
- `Daya Beli (/daya-beli/)`: simulasi per provinsi berbasis inferensi penuh Ridge Regression untuk pengeluaran riil per kapita

Tambahan halaman pendukung:

- `Panduan (/panduan/)`
- `Datasets (/datasets/)`
- `Compare (/compare/)`
- `Map (/map/)`
- `Scenarios (/scenarios/)`

## Forecasting inflasi multi-horizon

### Horizon publik

Pipeline forecast publik mendukung empat horizon:

- `1m`
- `3m`
- `6m`
- `12m`

Setiap horizon memiliki evaluasi, pemeringkatan model, dan artefak terpisah. Dengan demikian, model terbaik untuk 1 bulan tidak diasumsikan sama dengan model terbaik untuk 12 bulan.

### Kandidat model

Kandidat model yang dievaluasi:

- `Naive`
- `ARIMA`
- `SARIMAX`
- `Prophet`
- `LSTM`
- `Bi-LSTM`
- `Ensemble` horizon-specific
- `GARCH` opsional, hanya dipakai jika asumsi dan dependensi mendukung

Catatan metodologis:

- `GARCH` tidak dipaksa tampil. Jika package `arch` tidak tersedia atau validasi asumsi tidak memadai, kandidat ini ditandai `skipped`.
- Model deep learning tetap dicatat pada tabel perbandingan, namun headline publik hanya menggunakan model dengan performa terbaik pada horizon terkait.

### Aturan seleksi model

Untuk setiap horizon:

- `MAE` dipakai sebagai metrik utama ranking
- `RMSE` dan `sMAPE` dipakai sebagai metrik pendamping
- UI publik hanya menampilkan `2 model terbaik`

Pendekatan ini menjaga fokus halaman forecasting tanpa menghilangkan transparansi proses seleksi.

### Confidence interval

Confidence interval tidak dibentuk dari angka tetap atau elemen dekoratif.

Band prediksi dibangun dari:

1. residual out-of-sample hasil walk-forward backtest
2. distribusi residual historis per model-horizon
3. penerapan quantile residual ke point forecast

Implikasi interpretatif:

- horizon `1M` lebih sesuai untuk pembacaan taktis
- horizon `3M` dan `6M` lebih sesuai untuk arah kebijakan
- horizon `12M` lebih sesuai untuk orientasi makro, bukan angka presisi

Semakin jauh horizon, semakin lebar band-nya. Itu perilaku yang diharapkan, bukan bug.

## Model proksi daya beli

Modul ini tidak memprediksi daya beli murni dalam pengertian teoritis yang ketat. Target yang dipakai adalah
`pengeluaran riil per kapita per bulan`, lalu hasilnya dibaca sebagai proksi daya beli karena lebih dekat ke ruang
belanja riil rumah tangga setelah penyesuaian inflasi.

Estimasi dilakukan dengan `Ridge Regression` menggunakan baseline wilayah terbaru.

Karakteristik implementasi:

- simulasi tidak lagi menggunakan aproksimasi linear demo
- request simulasi membaca baseline wilayah terbaru dari `clean_daya_beli.csv`
- input user dioverride ke baseline tersebut
- fitur turunan dihitung ulang sebelum inferensi

Endpoint simulasi utama:

- `GET /api/simulate/?provinsi=...&inflasi=...`
- override opsional:
  - `ump`
  - `tpt`
  - `pdrb_hargakonstan`

Catatan interpretasi:

- `R^2` pada model proksi daya beli adalah skor goodness-of-fit pada data uji
- `R^2` bukan akurasi klasifikasi
- input `inflasi` pada simulasi dibaca sebagai skenario inflasi tahunan, lalu dipetakan ke skala fitur internal model secara konsisten

## Artefak utama

Artefak forecast utama:

- `models/inflation_multihorizon_forecast.json`
- `models/inflation_multihorizon_comparison.json`
- `models/forecast_results.json` sebagai payload turunan yang ikut disegarkan

Payload artefak forecast mencakup:

- `generated_at`
- `history`
- `horizons`
- `headline_model`
- `headline_forecast`
- `top_models`
- `risk_note`
- metrik model
- confidence interval

Artefak model lain yang masih dipakai:

- `models/best_daya_beli_ridge.pkl`
- `models/lstm_model.pt`
- `models/arima_inflasi.pkl`
- `models/ensemble_forecast.pkl`
- `models/ensemble_metrics.pkl`

## Endpoint penting

Sumber tunggal forecast UI:

- `GET /api/inflation-forecast/`

Endpoint pendukung:

- `GET /api/inflasi-summary/`
- `GET /api/usd-idr/`
- `GET /api/simulate/`
- `GET /api/commodity-prices/`
- `GET /api/metrics-latest/`

Aturan sinkronisasi UI:

- home, dashboard, dan halaman forecasting membaca headline inflasi dari payload yang sama
- headline publik default memakai `horizons["1m"]`
- model headline mengikuti `headline_model`, bukan hardcoded `LSTM` atau `Ensemble`

## Struktur file penting

```text
dashboard/
  manage.py
  predictions/
    templates/predictions/
    static/predictions/
    inflation_forecast.py
    views.py
  train_inflation_multihorizon.py

datasets/
  processed/
    clean_inflasi_ts.csv
    clean_daya_beli.csv

models/
  inflation_multihorizon_forecast.json
  inflation_multihorizon_comparison.json
  forecast_results.json
  best_daya_beli_ridge.pkl
```

## Retrain dan menjalankan proyek

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Refresh data

```bash
python download_domestic.py
python preprocessing.py
```

### 3. Retrain model proksi daya beli

```bash
python dashboard/train_daya_beli_ridge.py
```

### 4. Retrain forecast inflasi multi-horizon

```bash
python dashboard/train_inflation_multihorizon.py
```

Script ini akan memperbarui:

- ranking model per horizon
- top-2 model per horizon
- point forecast
- confidence interval
- artefak JSON yang dibaca web

### 5. Jalankan aplikasi

```bash
python dashboard/manage.py runserver
```

### 6. Jalankan test utama

```bash
python dashboard/manage.py test predictions.tests
```

## Catatan interpretasi

- horizon `1M` paling relevan untuk headline publik jangka dekat
- horizon `3M` dan `6M` lebih tepat untuk pembacaan arah tren
- horizon `12M` dipakai untuk konteks makro dan diskusi risiko
- band prediksi harus dibaca sebagai rentang estimasi, bukan jaminan realisasi

## Workflow branch tim

Branch kerja aktif untuk integrasi UI dan backend ringan:

- `frontend`

Alur integrasi:

1. kerjakan perubahan di `frontend`
2. audit dan test
3. push `frontend`
4. merge `frontend -> main`
5. sync `backend` dan `modelling` ke `main` terbaru

## Sinkronisasi branch tim

Jika branch lokal belum memiliki commit yang belum dipublikasikan:

```bash
git fetch origin
git checkout <branch-yang-dia-pakai>
git pull --ff-only origin <branch-yang-dia-pakai>
```

Jika branch lokal sudah memiliki commit sendiri dan perlu disejajarkan ke `main` terbaru:

```bash
git fetch origin
git checkout <branch-lokal>
git rebase origin/main
```

Jadi urutannya memang dimulai dari `git fetch origin`, lalu lanjut `pull --ff-only` atau `rebase` sesuai kondisi branch lokalnya.

## Anggota tim

- Muhammad Rajif Al Farikhi - Backend
- Sahrul Adicandra Effendy - Backend + Data
- Semaya David Petroes Putra - Modelling
- Adrina Firda Marwah - Modelling
- Okan Athallah Maredith - Frontend

