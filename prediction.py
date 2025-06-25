# prediction.py

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
            "month_cos": np.cos(2 * np.pi * df["Months"] / 12)
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

    raw = load_raw(data_path)
    if raw is None:
        st.error("CSV not found. Upload to continue.")
        return

    wells = [c for c in raw.columns if c.startswith("W")]
    model = st.radio("Choose Model", ["🔮 ANN", "📈 ARIMA"], horizontal=True)

    if model == "🔮 ANN":
        st.subheader("🔍 ANN Model Metrics (All Wells)")
        lags = st.sidebar.slider("Lag steps", 1, 24, 12)
        layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())
        scaler_choice = st.sidebar.selectbox("Scaler", ["Standard", "Robust"])

        metrics_table = []
        forecast_table = []

        for well in wells:
            try:
                clean = clean_series(raw, well)
                if len(clean) < lags * 5:
                    continue
                lo, hi = clip_bounds(clean[well])
                feat = add_lags(clean, well, lags)
                metrics, hist, future = train_ann(feat, well, layers, lags, scaler_choice, lo, hi)

                metrics_table.append({
                    "Well": well,
                    **metrics
                })

                future["Year"] = future["Date"].dt.year
                annual_avg = future.groupby("Year")["Depth"].mean().round(2)
                row = {"Well": well}
                for y in range(2025, 2030):
                    row[str(y)] = annual_avg.get(y, np.nan)
                forecast_table.append(row)

            except Exception:
                continue

        st.dataframe(pd.DataFrame(metrics_table), use_container_width=True)
        st.subheader("🗒️ ANN Forecast — Avg Depth per Year (2025–2029)")
        st.dataframe(pd.DataFrame(forecast_table), use_container_width=True)

    elif model == "📈 ARIMA":
        st.subheader("📋 ARIMA Metrics & 5-Year Forecast (All Wells)")

        arima_metrics = []
        forecast_rows = []

        for well in wells:
            try:
                df = raw[["Date", well]].dropna()
                df.set_index("Date", inplace=True)
                series = df[well]
                if len(series) < 24:
                    continue

                lo, hi = clip_bounds(series)
                train_size = int(len(series) * 0.8)
                train = series[:train_size]
                test = series[train_size:]

                model = ARIMA(train, order=(1, 1, 1)).fit()
                rmse = round(np.sqrt(mean_squared_error(test, model.forecast(len(test)))), 4)

                full_model = ARIMA(series, order=(1, 1, 1)).fit()
                future = full_model.get_forecast(60)
                future_values = np.clip(future.predicted_mean.values, lo, hi)
                future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=60, freq="MS")
                forecast_df = pd.DataFrame({"Date": future_dates, "Depth": future_values})
                forecast_df["Year"] = forecast_df["Date"].dt.year

                yearly_avg = forecast_df.groupby("Year")["Depth"].mean().round(2)

                arima_metrics.append({
                    "Well": well,
                    "AIC": round(full_model.aic, 1),
                    "BIC": round(full_model.bic, 1),
                    "RMSE Test": rmse
                })

                row = {"Well": well}
                for y in range(2025, 2030):
                    row[str(y)] = yearly_avg.get(y, np.nan)
                forecast_rows.append(row)

            except Exception:
                continue

        st.markdown("### 📈 ARIMA Model Metrics (All Wells)")
        st.dataframe(pd.DataFrame(arima_metrics), use_container_width=True)

        st.markdown("### 📅 ARIMA Forecast: Avg Depth per Year (2025–2029)")
        st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True)
