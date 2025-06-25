import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import plotly.express as px


def groundwater_prediction_page(data_path="GW_data_annual.csv"):
    st.title("📊 Groundwater Forecasting")

    HORIZON_M = 60  # 5 years

    @st.cache_data(show_spinner=False)
    def load_raw(path):
        if not Path(path).exists():
            return None
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "Date" not in df.columns:
            st.error(f"CSV file must contain a 'Date' column. Found columns: {list(df.columns)}")
            raise ValueError("Missing 'Date' column")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Months"] = df["Date"].dt.month
        return df.sort_values("Date").reset_index(drop=True)

    def clean_series(df, well):
        s = df[well].copy()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        s = s.where(s.between(q1 - 3 * iqr, q3 + 3 * iqr)).interpolate(limit_direction="both")
        out = pd.DataFrame({
            "Date": df["Date"], well: s,
            "Months": df["Months"],
            "month_sin": np.sin(2 * np.pi * df["Months"] / 12),
            "month_cos": np.cos(2 * np.pi * df["Months"] / 12),
            "Precipitation": df["Precipitation"]
        })
        return out.dropna().reset_index(drop=True)

    def add_lags(df, well, n):
        out = df.copy()
        for k in range(1, n + 1):
            out[f"{well}_lag{k}"] = out[well].shift(k)
        return out.dropna().reset_index(drop=True)

    def clip_bounds(series):
        lo, hi = series.min(), series.max()
        rng = hi - lo if hi > lo else max(hi, 1)
        return max(0, lo - 0.2 * rng), hi + 0.2 * rng

    def train_ann(df_feat, well, layers, lags, scaler_type, lo, hi):
        X = df_feat.drop(columns=[well, "Date"])
        y = df_feat[well]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = RobustScaler() if scaler_type == "Robust" else StandardScaler()
        mdl = MLPRegressor(hidden_layer_sizes=layers, max_iter=2000,
                           random_state=42, early_stopping=True)
        mdl.fit(scaler.fit_transform(Xtr), ytr)
        ytr_pred = np.clip(mdl.predict(scaler.transform(Xtr)), lo, hi)
        yte_pred = np.clip(mdl.predict(scaler.transform(Xte)), lo, hi)
        df_feat.loc[Xtr.index, "pred"] = ytr_pred
        df_feat.loc[Xte.index, "pred"] = yte_pred

        metrics = {"R² train": round(r2_score(ytr, ytr_pred), 4),
                   "RMSE train": round(np.sqrt(mean_squared_error(ytr, ytr_pred)), 4),
                   "R² test": round(r2_score(yte, yte_pred), 4),
                   "RMSE test": round(np.sqrt(mean_squared_error(yte, yte_pred)), 4)}

        feats = scaler.feature_names_in_
        r = df_feat.tail(1).iloc[0].copy()
        fut = []
        for _ in range(HORIZON_M):
            for k in range(lags, 1, -1):
                r[f"{well}_lag{k}"] = r[f"{well}_lag{k - 1}"]
            r[f"{well}_lag1"] = r["pred"]
            nxt = r["Date"] + pd.DateOffset(months=1)
            r.update({"Date": nxt, "Months": nxt.month,
                      "month_sin": np.sin(2 * np.pi * nxt.month / 12),
                      "month_cos": np.cos(2 * np.pi * nxt.month / 12)})
            val = np.clip(mdl.predict(scaler.transform(r[feats].to_frame().T))[0], lo, hi)
            r[well] = r["pred"] = val
            fut.append({"Date": nxt, "Depth": val})
        return metrics, df_feat, pd.DataFrame(fut)

    # The rest of the script remains unchanged...
    # Use the existing app logic for sidebar interaction, plotting, and results display.
    # This change ensures only 'Precipitation' is used from meteorological variables.
