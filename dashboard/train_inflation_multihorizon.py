import json
import logging
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from predictions.inflation_forecast import (
    CORE_EXOG_COLUMNS,
    FORECAST_HISTORY_WINDOW,
    FORECAST_INTERVAL_LEVEL,
    FORECAST_TEST_WINDOW,
    FORECAST_HORIZONS,
    SARIMAX_REGRESSOR_SHORTLIST,
    comparison_artifact_path,
    forecast_artifact_path,
    sarimax_feature_audit_path,
    label_for_horizon,
    make_forecast_payload,
    prepare_inflation_dataframe,
    professional_model_name,
    risk_note_for_horizon,
)


warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "processed", "clean_inflasi_ts.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ARIMA_ORDER = (3, 0, 3)
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 0, 12)
SEQ_LENGTH = 12
LSTM_EPOCHS = 80
LSTM_PATIENCE = 8
LSTM_HIDDEN = 48
LSTM_LR = 0.001
INTERVAL_ALPHA = (1.0 - FORECAST_INTERVAL_LEVEL) / 2.0
PROPHET_REGRESSOR_CANDIDATES = [c for c in CORE_EXOG_COLUMNS if c in set(SARIMAX_REGRESSOR_SHORTLIST)]


def smape(y_true, y_pred):
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denominator = np.abs(true) + np.abs(pred)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return float(np.mean(2.0 * np.abs(pred - true) / denominator) * 100.0)


def metric_block(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": float(smape(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def empirical_interval(point_forecast, residuals):
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if residuals.size < 8:
        return float(point_forecast), float(point_forecast), "confidence band terbatas"
    lower_offset = np.quantile(residuals, INTERVAL_ALPHA)
    upper_offset = np.quantile(residuals, 1.0 - INTERVAL_ALPHA)
    return (
        float(point_forecast + lower_offset),
        float(point_forecast + upper_offset),
        None,
    )


def get_feature_columns(df):
    excluded = {"Tanggal", "Inflasi_MoM"}
    return [c for c in df.columns if c not in excluded]


def get_prophet_regressors(df):
    return [column for column in PROPHET_REGRESSOR_CANDIDATES if column in df.columns]


def evaluate_naive(df, horizon):
    usable = df.iloc[:-horizon].copy()
    test = usable.tail(FORECAST_TEST_WINDOW).copy()
    y_true = df["Inflasi_MoM"].shift(-horizon).dropna().tail(FORECAST_TEST_WINDOW).values
    y_pred = test["Inflasi_MoM"].values
    point_forecast = float(df["Inflasi_MoM"].iloc[-1])
    return {
        "id": "naive",
        "name": professional_model_name("naive"),
        "metrics": metric_block(y_true, y_pred),
        "residuals": (y_true - y_pred).tolist(),
        "point_forecast": point_forecast,
        "metric_source": "walk_forward",
        "status": "ok",
    }


def build_future_exog(df, horizon, columns):
    future_dates = pd.date_range(
        df["Tanggal"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    last_row = df.iloc[-1]
    rows = []
    for future_date in future_dates:
        row = {column: float(last_row[column]) for column in columns}
        if "Bulan_Sin" in columns or "Bulan_Cos" in columns:
            month = future_date.month
            row["Bulan_Sin"] = float(np.sin(2 * np.pi * month / 12.0))
            row["Bulan_Cos"] = float(np.cos(2 * np.pi * month / 12.0))
        if "Oil_x_USDIDR" in columns and "Harga_Minyak_USD" in row and "USD_IDR" in row:
            row["Oil_x_USDIDR"] = float(row["Harga_Minyak_USD"] * row["USD_IDR"])
        rows.append(row)
    return pd.DataFrame(rows), future_dates


def walkforward_arima(df, horizon):
    y = df["Inflasi_MoM"].reset_index(drop=True)
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_y = y.iloc[: origin + 1]
        actual = float(y.iloc[origin + horizon])
        try:
            model = ARIMA(train_y, order=ARIMA_ORDER)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)
            pred = float(np.asarray(forecast).reshape(-1)[-1])
        except Exception:
            pred = float(train_y.iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    full_model = ARIMA(y, order=ARIMA_ORDER).fit()
    future_point = float(np.asarray(full_model.forecast(steps=horizon)).reshape(-1)[-1])
    return {
        "id": "arima",
        "name": professional_model_name("arima"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d") for origin in range(start, len(df) - horizon)],
    }


def walkforward_sarimax(df, horizon, regressors=None, result_id="sarimax", result_name=None):
    y = df["Inflasi_MoM"].reset_index(drop=True)
    regressors = [
        column for column in (regressors if regressors is not None else get_prophet_regressors(df))
        if column in df.columns
    ]
    exog = df[regressors].reset_index(drop=True) if regressors else None
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_y = y.iloc[: origin + 1]
        train_exog = exog.iloc[: origin + 1] if exog is not None else None
        future_exog = exog.iloc[origin + 1 : origin + horizon + 1] if exog is not None else None
        actual = float(y.iloc[origin + horizon])
        try:
            model = SARIMAX(
                train_y,
                exog=train_exog,
                order=SARIMAX_ORDER,
                seasonal_order=SARIMAX_SEASONAL_ORDER,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=horizon, exog=future_exog)
            pred = float(np.asarray(forecast).reshape(-1)[-1])
        except Exception:
            pred = float(train_y.iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    future_exog, _ = build_future_exog(df, horizon, regressors) if regressors else (None, None)
    full_model = SARIMAX(
        y,
        exog=exog,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    future_point = float(np.asarray(full_model.forecast(steps=horizon, exog=future_exog)).reshape(-1)[-1])
    return {
        "id": result_id,
        "name": result_name or professional_model_name("sarimax"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "regressors": regressors,
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d") for origin in range(start, len(df) - horizon)],
    }


def build_sarimax_feature_audit(df, horizon, base_result):
    regressors = list(base_result.get("regressors") or get_prophet_regressors(df))
    base_mae = float(base_result["metrics"]["mae"])
    audit_rows = []

    for feature in regressors:
        remaining = [column for column in regressors if column != feature]
        try:
            reduced_result = walkforward_sarimax(
                df,
                horizon,
                regressors=remaining,
                result_id="sarimax_ablation",
                result_name="SARIMAX drop-one",
            )
            reduced_mae = float(reduced_result["metrics"]["mae"])
            delta_mae = reduced_mae - base_mae
            audit_rows.append(
                {
                    "feature": feature,
                    "remaining_regressors": remaining,
                    "status": "ok",
                    "dropped_model_metrics": {
                        "mae": round(reduced_mae, 4),
                        "rmse": round(float(reduced_result["metrics"]["rmse"]), 4),
                        "smape": round(float(reduced_result["metrics"]["smape"]), 2),
                        "n_test": int(reduced_result["metrics"]["n_test"]),
                    },
                    "delta_mae": round(delta_mae, 4),
                    "delta_rmse": round(float(reduced_result["metrics"]["rmse"] - base_result["metrics"]["rmse"]), 4),
                    "interpretation": (
                        "penghapusan fitur memperburuk error; fitur memberi kontribusi positif"
                        if delta_mae > 0.01
                        else "penghapusan fitur hampir tidak mengubah error; kontribusi fitur cenderung marjinal"
                        if delta_mae >= -0.01
                        else "penghapusan fitur justru menurunkan error; shortlist layak ditinjau ulang"
                    ),
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "feature": feature,
                    "remaining_regressors": remaining,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )

    audit_rows.sort(
        key=lambda item: (
            item.get("status") != "ok",
            -item.get("delta_mae", -9999),
            item.get("feature", ""),
        )
    )
    return {
        "base_regressors": regressors,
        "base_metrics": {
            "mae": round(base_mae, 4),
            "rmse": round(float(base_result["metrics"]["rmse"]), 4),
            "smape": round(float(base_result["metrics"]["smape"]), 2),
            "n_test": int(base_result["metrics"]["n_test"]),
        },
        "method": "drop-one walk-forward ablation",
        "drop_one_tests": audit_rows,
    }


def _fit_prophet(train_frame):
    regressors = get_prophet_regressors(train_frame)
    prophet_df = train_frame[["Tanggal", "Inflasi_MoM"] + regressors].rename(
        columns={"Tanggal": "ds", "Inflasi_MoM": "y"}
    )
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
    )
    for regressor in regressors:
        model.add_regressor(regressor)
    model.fit(prophet_df)
    return model, regressors


def walkforward_prophet(df, horizon):
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_frame = df.iloc[: origin + 1].copy()
        future_frame = df.iloc[origin + 1 : origin + horizon + 1].copy()
        actual = float(df["Inflasi_MoM"].iloc[origin + horizon])
        try:
            model, regressors = _fit_prophet(train_frame)
            predict_df = future_frame[["Tanggal"] + regressors].rename(columns={"Tanggal": "ds"})
            forecast = model.predict(predict_df)
            pred = float(forecast["yhat"].iloc[-1])
        except Exception:
            pred = float(train_frame["Inflasi_MoM"].iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    model, regressors = _fit_prophet(df.copy())
    future_exog, future_dates = build_future_exog(df, horizon, regressors)
    future_frame = future_exog.copy()
    future_frame.insert(0, "ds", future_dates)
    future_point = float(model.predict(future_frame)["yhat"].iloc[-1])
    return {
        "id": "prophet",
        "name": professional_model_name("prophet"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d") for origin in range(start, len(df) - horizon)],
    }


class SequenceForecastModel(nn.Module):
    def __init__(self, input_size, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(LSTM_HIDDEN * direction_factor, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        tail = output[:, -1, :]
        return self.fc(self.dropout(tail))


def _build_sequence_supervised(df, horizon, feature_columns):
    target = df["Inflasi_MoM"].shift(-horizon)
    usable = df.copy()
    usable["target"] = target
    usable = usable.dropna(subset=["target"]).reset_index(drop=True)

    feature_values = usable[feature_columns].values
    target_values = usable["target"].values.reshape(-1, 1)
    dates = usable["Tanggal"].tolist()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(feature_values)
    y_scaled = scaler_y.fit_transform(target_values)

    sequences = []
    targets = []
    sequence_dates = []
    for idx in range(SEQ_LENGTH - 1, len(usable)):
        start = idx - SEQ_LENGTH + 1
        sequences.append(x_scaled[start : idx + 1])
        targets.append(y_scaled[idx])
        sequence_dates.append(dates[idx])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        sequence_dates,
        scaler_x,
        scaler_y,
        x_scaled,
    )


def _train_sequence_model(X_train, y_train, X_val, y_val, bidirectional=False):
    device = torch.device("cpu")
    model = SequenceForecastModel(X_train.shape[2], bidirectional=bidirectional).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    patience_left = LSTM_PATIENCE

    for _ in range(LSTM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            patience_left = LSTM_PATIENCE
            best_state = deepcopy(model.state_dict())
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_sequence_model(df, horizon, model_id):
    bidirectional = model_id == "bilstm"
    feature_columns = get_feature_columns(df)
    (
        X_seq,
        y_seq,
        sequence_dates,
        scaler_x,
        scaler_y,
        x_scaled_all,
    ) = _build_sequence_supervised(df, horizon, feature_columns)

    if len(X_seq) <= FORECAST_TEST_WINDOW + 12:
        raise RuntimeError("Sequence dataset terlalu kecil untuk evaluasi.")

    test_size = FORECAST_TEST_WINDOW
    val_size = 12
    train_end = len(X_seq) - test_size - val_size
    val_end = len(X_seq) - test_size

    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]
    X_val = X_seq[train_end:val_end]
    y_val = y_seq[train_end:val_end]
    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]

    model = _train_sequence_model(X_train, y_train, X_val, y_val, bidirectional=bidirectional)
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
    test_pred = scaler_y.inverse_transform(test_pred_scaled).reshape(-1)
    y_true = scaler_y.inverse_transform(y_test).reshape(-1)

    last_sequence = x_scaled_all[-SEQ_LENGTH:]
    with torch.no_grad():
        future_scaled = model(torch.tensor(np.array([last_sequence]), dtype=torch.float32)).cpu().numpy()
    point_forecast = float(scaler_y.inverse_transform(future_scaled).reshape(-1)[0])

    return {
        "id": model_id,
        "name": professional_model_name(model_id),
        "metrics": metric_block(y_true, test_pred),
        "residuals": (y_true - test_pred).tolist(),
        "point_forecast": point_forecast,
        "metric_source": "chronological_holdout",
        "status": "ok",
        "backtest_dates": [dt.strftime("%Y-%m-%d") for dt in sequence_dates[val_end:]],
    }


def maybe_garch_candidate():
    return {
        "id": "garch",
        "name": professional_model_name("garch"),
        "status": "skipped",
        "reason": "Package arch tidak tersedia, sehingga kandidat GARCH dilewati secara eksplisit.",
        "metric_source": "not_evaluated",
    }


def build_ensemble_result(base_results):
    usable = [result for result in base_results if result.get("status") == "ok"]
    if len(usable) < 2:
        return None

    weights_raw = {
        result["id"]: 1.0 / max(result["metrics"]["mae"], 1e-6)
        for result in usable
    }
    weight_total = sum(weights_raw.values())
    weights = {key: value / weight_total for key, value in weights_raw.items()}

    prediction_matrix = np.vstack(
        [np.asarray(result["backtest_predictions"], dtype=float) for result in usable]
    )
    actuals = np.asarray(usable[0]["backtest_actuals"], dtype=float)
    ensemble_pred = np.zeros_like(actuals)
    for idx, result in enumerate(usable):
        ensemble_pred += weights[result["id"]] * prediction_matrix[idx]

    future_point = 0.0
    for result in usable:
        future_point += weights[result["id"]] * float(result["point_forecast"])

    return {
        "id": "ensemble",
        "name": professional_model_name("ensemble"),
        "metrics": metric_block(actuals, ensemble_pred),
        "residuals": (actuals - ensemble_pred).tolist(),
        "point_forecast": float(future_point),
        "metric_source": "walk_forward",
        "status": "ok",
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "backtest_predictions": ensemble_pred.tolist(),
        "backtest_actuals": actuals.tolist(),
        "backtest_dates": usable[0]["backtest_dates"],
    }
def summarize_candidate(result):
    summary = {
        "id": result["id"],
        "name": result["name"],
        "status": result.get("status", "ok"),
        "metric_source": result.get("metric_source", "walk_forward"),
    }
    if result.get("status") == "ok":
        summary["metrics"] = {
            "mae": round(float(result["metrics"]["mae"]), 4),
            "rmse": round(float(result["metrics"]["rmse"]), 4),
            "smape": round(float(result["metrics"]["smape"]), 2),
            "n_test": int(result["metrics"]["n_test"]),
        }
        summary["point_forecast"] = round(float(result["point_forecast"]), 4)
    if "reason" in result:
        summary["reason"] = result["reason"]
    if "weights" in result:
        summary["weights"] = result["weights"]
    return summary


def forecast_for_horizon(df, horizon):
    print(f"\n=== Horizon {horizon} bulan ===")
    last_date = df["Tanggal"].iloc[-1]
    future_date = (last_date + pd.DateOffset(months=horizon)).strftime("%Y-%m")

    naive_result = evaluate_naive(df, horizon)
    arima_result = walkforward_arima(df, horizon)
    sarimax_result = walkforward_sarimax(df, horizon)
    prophet_result = walkforward_prophet(df, horizon)

    deep_candidates = []
    for model_id in ("lstm", "bilstm"):
        try:
            deep_candidates.append(evaluate_sequence_model(df, horizon, model_id))
        except Exception as exc:
            deep_candidates.append(
                {
                    "id": model_id,
                    "name": professional_model_name(model_id),
                    "status": "skipped",
                    "reason": str(exc),
                    "metric_source": "chronological_holdout",
                }
            )

    garch_candidate = maybe_garch_candidate()

    base_results = [naive_result, arima_result, sarimax_result, prophet_result]
    ensemble_result = build_ensemble_result([arima_result, sarimax_result, prophet_result])
    if ensemble_result is not None:
        base_results.append(ensemble_result)

    ranked_public = sorted(
        [result for result in base_results if result.get("status") == "ok"],
        key=lambda item: item["metrics"]["mae"],
    )[:2]

    top_models = []
    for rank, result in enumerate(ranked_public, start=1):
        ci_lower, ci_upper, interval_note = empirical_interval(result["point_forecast"], result["residuals"])
        top_models.append(
            {
                "id": result["id"],
                "name": result["name"],
                "rank": rank,
                "point_forecast": round(float(result["point_forecast"]), 4),
                "ci_lower": round(float(ci_lower), 4),
                "ci_upper": round(float(ci_upper), 4),
                "metrics": {
                    "mae": round(float(result["metrics"]["mae"]), 4),
                    "rmse": round(float(result["metrics"]["rmse"]), 4),
                    "smape": round(float(result["metrics"]["smape"]), 2),
                    "n_test": int(result["metrics"]["n_test"]),
                    "interval_level": int(FORECAST_INTERVAL_LEVEL * 100),
                },
                "series": {
                    "forecast_label": future_date,
                    "anchor_label": df["Tanggal"].iloc[-1].strftime("%Y-%m"),
                    "anchor_actual": round(float(df["Inflasi_MoM"].iloc[-1]), 4),
                },
                "interval_note": interval_note,
            }
        )

    comparison = [summarize_candidate(result) for result in base_results]
    comparison.extend(summarize_candidate(result) for result in deep_candidates)
    comparison.append(summarize_candidate(garch_candidate))
    comparison = sorted(
        comparison,
        key=lambda item: (
            item["status"] != "ok",
            item.get("metrics", {}).get("mae", 9999.0),
            item["name"],
        ),
    )

    return {
        "label": label_for_horizon(horizon),
        "forecast_months": horizon,
        "forecast_date": future_date,
        "headline_model": top_models[0]["id"] if top_models else None,
        "headline_forecast": top_models[0]["point_forecast"] if top_models else None,
        "headline_interval": {
            "lower": top_models[0]["ci_lower"] if top_models else None,
            "upper": top_models[0]["ci_upper"] if top_models else None,
            "level": int(FORECAST_INTERVAL_LEVEL * 100),
        },
        "future_labels": [future_date],
        "top_models": top_models,
        "series": {
            "history_labels": df["Tanggal"].tail(FORECAST_HISTORY_WINDOW).dt.strftime("%Y-%m").tolist(),
            "history_actual": [round(float(v), 4) for v in df["Inflasi_MoM"].tail(FORECAST_HISTORY_WINDOW).tolist()],
        },
        "risk_note": risk_note_for_horizon(horizon),
        "comparison": comparison,
        "sarimax_feature_audit": build_sarimax_feature_audit(df, horizon, sarimax_result),
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = prepare_inflation_dataframe(df)

    horizon_results = {}
    comparison_summary = {}
    sarimax_feature_audit = {
        "generated_at": None,
        "methodology": {
            "selection_basis": "Shortlist awal ditentukan dari teori ekonomi, lalu diuji ulang dengan ablation drop-one out-of-sample pada SARIMAX.",
            "metric_primary": "MAE",
            "note": "Delta MAE positif berarti menghapus fitur memperburuk performa, sehingga fitur tersebut membantu model."
        },
        "horizons": {},
    }
    for horizon_key, horizon_months in FORECAST_HORIZONS.items():
        result = forecast_for_horizon(df, horizon_months)
        horizon_results[horizon_key] = result
        comparison_summary[horizon_key] = result["comparison"]
        sarimax_feature_audit["horizons"][horizon_key] = {
            "label": result["label"],
            "forecast_months": horizon_months,
            **(result.get("sarimax_feature_audit") or {}),
        }

    payload = make_forecast_payload(df, horizon_results, comparison_summary)
    sarimax_feature_audit["generated_at"] = payload.get("generated_at")

    forecast_path = forecast_artifact_path(PROJECT_ROOT)
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    comparison_path = comparison_artifact_path(PROJECT_ROOT)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)

    feature_audit_path = sarimax_feature_audit_path(PROJECT_ROOT)
    with open(feature_audit_path, "w", encoding="utf-8") as f:
        json.dump(sarimax_feature_audit, f, ensure_ascii=False, indent=2)

    # Keep the historical filename in sync so tracked artifacts still tell the latest story.
    with open(os.path.join(MODELS_DIR, "forecast_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {forecast_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {feature_audit_path}")
    print(f"Saved: {os.path.join(MODELS_DIR, 'forecast_results.json')}")
    return payload


if __name__ == "__main__":
    main()
