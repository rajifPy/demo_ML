import os
import pickle
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictions.daya_beli_model import (
    MODEL_CATEGORICAL_FEATURES,
    RAW_COMPONENT_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABEL,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    YEAR_COLUMN,
    build_model_frame,
)


ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]
FEATURE_SELECTION_TOLERANCE = 1500.0
MIN_FEATURE_COUNT = 8


@dataclass
class MetricSummary:
    r2: float
    mae: float
    rmse: float
    smape: float


def metric_summary(y_true, y_pred) -> MetricSummary:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    smape = np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred_arr - y_true_arr) / denom)
    return MetricSummary(
        r2=float(r2_score(y_true_arr, y_pred_arr)),
        mae=float(mean_absolute_error(y_true_arr, y_pred_arr)),
        rmse=float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        smape=float(np.mean(smape)),
    )


def build_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", Ridge()),
        ]
    )


def iter_expanding_year_folds(df, min_train_years=1):
    unique_years = sorted(df[YEAR_COLUMN].unique().tolist())
    for idx in range(min_train_years, len(unique_years)):
        train_years = unique_years[:idx]
        validation_year = unique_years[idx]
        train_df = df[df[YEAR_COLUMN].isin(train_years)].copy()
        validation_df = df[df[YEAR_COLUMN] == validation_year].copy()
        if train_df.empty or validation_df.empty:
            continue
        yield train_years, validation_year, train_df, validation_df


def evaluate_year_folds(df, numeric_features, categorical_features, alpha, min_train_years=1):
    folds = []
    feature_columns = numeric_features + categorical_features

    for train_years, validation_year, train_df, validation_df in iter_expanding_year_folds(
        df,
        min_train_years=min_train_years,
    ):
        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_validation = validation_df[feature_columns]
        y_validation = validation_df[TARGET_COLUMN]

        model = build_pipeline(numeric_features, categorical_features)
        model.set_params(regressor__alpha=alpha)
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        metrics = metric_summary(y_validation, predictions)
        folds.append(
            {
                "train_years": [int(year) for year in train_years],
                "validation_year": int(validation_year),
                **asdict(metrics),
            }
        )

    if not folds:
        return {"folds": [], "mean": None}

    return {
        "folds": folds,
        "mean": {
            "mae": float(np.mean([fold["mae"] for fold in folds])),
            "rmse": float(np.mean([fold["rmse"] for fold in folds])),
            "r2": float(np.mean([fold["r2"] for fold in folds])),
            "smape": float(np.mean([fold["smape"] for fold in folds])),
        },
    }


def select_best_alpha(train_df, numeric_features, categorical_features):
    trials = []
    for alpha in ALPHA_GRID:
        evaluation = evaluate_year_folds(
            train_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            alpha=alpha,
        )
        mean_metrics = evaluation["mean"]
        if mean_metrics is None:
            continue
        trials.append(
            {
                "alpha": float(alpha),
                "mean": mean_metrics,
                "folds": evaluation["folds"],
            }
        )

    if not trials:
        raise RuntimeError("Tidak ada fold valid untuk pemilihan alpha time-aware.")

    best_trial = min(
        trials,
        key=lambda item: (
            item["mean"]["mae"],
            item["mean"]["rmse"],
            -item["mean"]["r2"],
        ),
    )
    return best_trial["alpha"], trials


def evaluate_feature_drop(train_df, numeric_features, categorical_features, alpha, base_mae):
    audit_rows = []
    for feature in numeric_features:
        reduced_features = [item for item in numeric_features if item != feature]
        evaluation = evaluate_year_folds(
            train_df,
            numeric_features=reduced_features,
            categorical_features=categorical_features,
            alpha=alpha,
        )
        mean_metrics = evaluation["mean"]
        if mean_metrics is None:
            continue
        delta_mae = float(mean_metrics["mae"] - base_mae)
        audit_rows.append(
            {
                "feature": feature,
                "dropped_mae": float(mean_metrics["mae"]),
                "delta_mae": delta_mae,
                "dropped_rmse": float(mean_metrics["rmse"]),
                "dropped_r2": float(mean_metrics["r2"]),
                "dropped_smape": float(mean_metrics["smape"]),
                "folds": evaluation["folds"],
                "decision_hint": (
                    "feature_membantu"
                    if delta_mae > FEATURE_SELECTION_TOLERANCE
                    else "feature_bisa_dilepas"
                    if delta_mae < -FEATURE_SELECTION_TOLERANCE
                    else "feature_marginal"
                ),
            }
        )
    return sorted(audit_rows, key=lambda item: item["delta_mae"])


def backward_select_features(train_df, numeric_features, categorical_features, alpha):
    selected = numeric_features.copy()
    selection_rounds = []

    while len(selected) > MIN_FEATURE_COUNT:
        evaluation = evaluate_year_folds(
            train_df,
            numeric_features=selected,
            categorical_features=categorical_features,
            alpha=alpha,
        )
        base_mae = evaluation["mean"]["mae"]
        audit_rows = evaluate_feature_drop(
            train_df,
            numeric_features=selected,
            categorical_features=categorical_features,
            alpha=alpha,
            base_mae=base_mae,
        )
        removable = next(
            (row for row in audit_rows if row["delta_mae"] < -FEATURE_SELECTION_TOLERANCE),
            None,
        )
        selection_rounds.append(
            {
                "active_features": selected.copy(),
                "base_metrics": evaluation["mean"],
                "drop_tests": audit_rows,
                "removed_feature": removable["feature"] if removable else None,
            }
        )
        if removable is None:
            break
        selected = [feature for feature in selected if feature != removable["feature"]]

    final_evaluation = evaluate_year_folds(
        train_df,
        numeric_features=selected,
        categorical_features=categorical_features,
        alpha=alpha,
    )
    return selected, selection_rounds, final_evaluation


def run_walk_forward(df, numeric_features, categorical_features, alpha):
    return evaluate_year_folds(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        alpha=alpha,
        min_train_years=1,
    )


def evaluate_by_province(model, test_df, feature_columns):
    frame = test_df.copy()
    frame["prediction"] = model.predict(frame[feature_columns])
    rows = []
    for province, province_df in frame.groupby("Provinsi"):
        y_true = province_df[TARGET_COLUMN].to_numpy(dtype=float)
        y_pred = province_df["prediction"].to_numpy(dtype=float)
        denom = np.abs(y_true) + np.abs(y_pred)
        smape = np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / denom)
        rows.append(
            {
                "province": province,
                "r2": None,
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "smape": float(np.mean(smape)),
                "n_rows": int(len(province_df)),
            }
        )
    return sorted(rows, key=lambda item: item["mae"], reverse=True)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "datasets", "processed", "clean_daya_beli.csv")
    output_path = os.path.join(project_root, "models", "best_daya_beli_ridge.pkl")

    df = pd.read_csv(dataset_path)
    model_df, numeric_features, categorical_features = build_model_frame(df)

    train_df = model_df[model_df[YEAR_COLUMN] <= TRAIN_END_YEAR].copy()
    test_df = model_df[model_df[YEAR_COLUMN] >= TEST_START_YEAR].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Split train/test kosong. Cek rentang tahun dataset.")

    best_alpha, alpha_trials = select_best_alpha(
        train_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    selected_features, selection_rounds, train_validation_summary = backward_select_features(
        train_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        alpha=best_alpha,
    )

    feature_columns = selected_features + categorical_features
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline(selected_features, categorical_features)
    pipeline.set_params(regressor__alpha=best_alpha)
    pipeline.fit(X_train, y_train)

    train_metrics = metric_summary(y_train, pipeline.predict(X_train))
    test_metrics = metric_summary(y_test, pipeline.predict(X_test))
    walk_forward = run_walk_forward(
        model_df,
        numeric_features=selected_features,
        categorical_features=categorical_features,
        alpha=best_alpha,
    )
    province_breakdown = evaluate_by_province(pipeline, test_df, feature_columns)

    bundle = {
        "pipeline": pipeline,
        "num_features": selected_features,
        "candidate_num_features": numeric_features,
        "cat_features": categorical_features,
        "target_column": TARGET_COLUMN,
        "target_label": TARGET_LABEL,
        "target_type": "real",
        "nominal_target_column": "Total_Pengeluaran",
        "excluded_raw_component_columns": RAW_COMPONENT_COLUMNS,
        "data_scope": {
            "year_min": int(model_df[YEAR_COLUMN].min()),
            "year_max": int(model_df[YEAR_COLUMN].max()),
            "province_count": int(model_df["Provinsi"].nunique()),
            "row_count": int(len(model_df)),
        },
        "split_strategy": {
            "type": "chronological_by_year",
            "train_end_year": TRAIN_END_YEAR,
            "test_start_year": TEST_START_YEAR,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_years": sorted(int(year) for year in train_df[YEAR_COLUMN].unique().tolist()),
            "test_years": sorted(int(year) for year in test_df[YEAR_COLUMN].unique().tolist()),
        },
        "validation_strategy": {
            "alpha_selection": "expanding_window_by_year",
            "feature_selection": "backward_elimination_by_validation_mae",
            "primary_metric": "mae",
            "feature_selection_tolerance": FEATURE_SELECTION_TOLERANCE,
            "alpha_trials": alpha_trials,
            "feature_selection_rounds": selection_rounds,
            "train_validation_summary": train_validation_summary,
        },
        "best_alpha": float(best_alpha),
        "train_r2": train_metrics.r2,
        "train_mae": train_metrics.mae,
        "train_rmse": train_metrics.rmse,
        "train_smape": train_metrics.smape,
        "test_r2": test_metrics.r2,
        "test_mae": test_metrics.mae,
        "test_rmse": test_metrics.rmse,
        "test_smape": test_metrics.smape,
        "walk_forward": walk_forward,
        "province_breakdown": province_breakdown,
        "model_note": (
            "Ridge regression untuk pengeluaran riil per kapita dengan target terdeflasi, "
            "alpha dipilih memakai expanding-window validation, dan feature set difilter via backward elimination."
        ),
    }

    with open(output_path, "wb") as file_obj:
        pickle.dump(bundle, file_obj)

    print("Saved:", output_path)
    print("Rows:", bundle["data_scope"]["row_count"], "| Provinces:", bundle["data_scope"]["province_count"])
    print("Years:", bundle["data_scope"]["year_min"], "-", bundle["data_scope"]["year_max"])
    print("Candidate features:", len(numeric_features), "| Selected features:", len(selected_features))
    print("Best alpha:", bundle["best_alpha"])
    print("Selected features:", ", ".join(selected_features))
    print("Train R2:", round(bundle["train_r2"], 4), "| Test R2:", round(bundle["test_r2"], 4))
    print("Test MAE:", round(bundle["test_mae"], 2), "| Test RMSE:", round(bundle["test_rmse"], 2))
    print("Test sMAPE:", round(bundle["test_smape"], 2))
    if bundle["walk_forward"]["mean"] is not None:
        print(
            "Walk-forward mean:",
            {
                "r2": round(bundle["walk_forward"]["mean"]["r2"], 4),
                "mae": round(bundle["walk_forward"]["mean"]["mae"], 2),
                "rmse": round(bundle["walk_forward"]["mean"]["rmse"], 2),
                "smape": round(bundle["walk_forward"]["mean"]["smape"], 2),
            },
        )


if __name__ == "__main__":
    main()
