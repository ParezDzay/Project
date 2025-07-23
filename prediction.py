import streamlit as st
import pandas as pd
import numpy as np
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
        # Prepare data
        X = df_feat.drop(columns=[well, "Date"])
        y = df_feat[well]

        n = len(df_feat)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        # Time-based split (no shuffle)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Choose scaler
        scaler = RobustScaler() if scaler_type == "Robust" else StandardScaler()

        # Fit scaler on train, transform train/val/test
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train MLPRegressor with early stopping, using internal validation fraction
        mdl = MLPRegressor(
            hidden_layer_sizes=layers,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=len(X_val) / (len(X_train) + len(X_val)),
            n_iter_no_change=20
        )

        mdl.fit(X_train_scaled, y_train)

        # Predict
        y_train_pred = np.clip(mdl.predict(X_train_scaled), lo, hi)
        y_val_pred = np.clip(mdl.predict(X_val_scaled), lo, hi)
        y_test_pred = np.clip(mdl.predict(X_test_scaled), lo, hi)

        # Update df_feat with predictions
        df_feat.loc[X_train.index, "pred"] = y_train_pred
        df_feat.loc[X_val.index, "pred"] = y_val_pred
        df_feat.loc[X_test.index, "pred"] = y_test_pred

        # Metrics on train and test (validation metrics could also be reported if needed)
        metrics = {
            "R² train": round(r2_score(y_train, y_train_pred), 4),
            "RMSE train": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
            "R² test": round(r2_score(y_test, y_test_pred), 4),
            "RMSE test": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
            "lags": lags,
            "layers": layers
        }

        # Forecast future 5 years monthly (60 months)
        r = df_feat.iloc[-1].copy()
        # Use column names of X as features
        feats = X.columns

        fut = []
        for _ in range(HORIZON_M):
            for k in range(lags, 1, -1):
                r[f"{well}_lag{k}"] = r[f"{well}_lag{k - 1}"]
            r[f"{well}_lag1"] = r["pred"]
            nxt = r["Date"] + pd.DateOffset(months=1)
            r.update({
                "Date": nxt,
                "Months": nxt.month,
                "month_sin": np.sin(2 * np.pi * nxt.month / 12),
                "month_cos": np.cos(2 * np.pi * nxt.month / 12)
            })
            val = np.clip(mdl.predict(scaler.transform(r[feats].to_frame().T))[0], lo, hi)
            r[well] = r["pred"] = val
            fut.append({"Date": nxt, "Depth": val})

        return metrics, df_feat, pd.DataFrame(fut)

    raw = load_raw(data_path)
    if raw is None:
        st.error("CSV not found. Upload to continue.")
        return

    wells = [c for c in raw.columns if c.startswith("W")]
    model = st.radio("Choose Model", ["🔮 ANN", "📉 ARIMA"], horizontal=True)

    if model == "🔮 ANN":
        # Initialize saved forecasts DataFrame with proper columns if not exist
        if "ann_results" not in st.session_state:
            st.session_state.ann_results = pd.DataFrame(columns=[
                "Well", "2025", "2026", "2027", "2028", "2029",
                "R² train", "RMSE train", "R² test", "RMSE test",
                "lags", "layers"
            ])

        well = st.sidebar.selectbox("Well", wells)
        clean = clean_series(raw, well)
        lo, hi = clip_bounds(clean[well])
        lags = st.sidebar.slider("Lag steps", 1, 24, 12)
        if len(clean) < lags * 10:
            lags = max(1, len(clean) // 10)
            st.info(f"Lags auto-reduced to {lags}")
        layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())
        scaler_choice = st.sidebar.selectbox("Scaler", ["Standard", "Robust"])
        feat = add_lags(clean, well, lags)
        metrics, hist, future = train_ann(feat, well, layers, lags, scaler_choice, lo, hi)

        st.subheader("🔍 ANN Model Metrics")
        st.json(metrics)

        df_act = pd.DataFrame({"Date": clean["Date"], "Depth": clean[well], "Type": "Actual"})
        df_fit = hist[["Date", "pred"]].rename(columns={"pred": "Depth"}).assign(Type="Predicted")
        df_for = future.assign(Type="Forecast")
        plot_df = pd.concat([df_act, df_fit, df_for])

        forecast_start = df_act["Date"].max()

        fig = px.line(
            plot_df,
            x="Date",
            y="Depth",
            color="Type",
            line_dash="Type",
            labels={"Depth": "Water-table depth (m)", "Date": "Date", "Type": "Legend"},
            title=f"{well} — ANN Fit & 5-Year Forecast",
            render_mode="svg",
            line_shape="spline"
        )

        fig.update_yaxes(autorange="reversed")

        fig.update_traces(
            selector=dict(name="Forecast"),
            line=dict(dash="dash", width=2),
            opacity=0.7
        )

        fig.add_shape(
            type="line",
            x0=forecast_start,
            x1=forecast_start,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", dash="dot")
        )

        fig.add_annotation(
            x=forecast_start,
            y=1,
            yref="paper",
            showarrow=False,
            text="Forecast Start",
            bgcolor="white",
            font=dict(color="gray")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🗒️ 5-Year Forecast Table (Annual Average)")
        df_for["Year"] = df_for["Date"].dt.year
        annual_forecast = df_for.groupby("Year")["Depth"].mean().reset_index()
        annual_forecast["Depth"] = annual_forecast["Depth"].round(2)
        st.dataframe(annual_forecast, use_container_width=True)

        # --------- Save & Download functionality ----------

        if st.button("➕ Save This Forecast"):
            row = {
                "Well": well,
                "R² train": metrics["R² train"],
                "RMSE train": metrics["RMSE train"],
                "R² test": metrics["R² test"],
                "RMSE test": metrics["RMSE test"],
                "lags": lags,
                "layers": ",".join(str(x) for x in layers)
            }
            for year in range(2025, 2030):
                val = annual_forecast.loc[annual_forecast["Year"] == year, "Depth"]
                row[str(year)] = val.values[0] if not val.empty else None

            # Remove old entry if exists and append new one
            st.session_state.ann_results = st.session_state.ann_results[st.session_state.ann_results["Well"] != well]
            st.session_state.ann_results = pd.concat([st.session_state.ann_results, pd.DataFrame([row])], ignore_index=True)

            st.success(f"Forecast for {well} saved.")

        if not st.session_state.ann_results.empty:
            st.markdown("### 📥 Saved ANN Forecasts and Metrics")
            st.dataframe(st.session_state.ann_results, use_container_width=True)

            csv = st.session_state.ann_results.to_csv(index=False).encode("utf-8")
            st.download_button("📁 Download All Saved Forecasts and Metrics as CSV", csv, "ANN_Forecasts_Metrics.csv", "text/csv")

        if st.button("🗑️ Clear All Saved Forecasts"):
            st.session_state.ann_results = st.session_state.ann_results.iloc[0:0]
            st.success("All saved forecasts cleared.")

    elif model == "📉 ARIMA":
        st.subheader("📋 ARIMA Forecast — Without Meteorological Variables")

        arima_metrics = []
        forecast_rows = []

        for well in wells:
            try:
                df = raw[["Date", well]].dropna()
                df.set_index("Date", inplace=True)
                series = df[well]

                if len(series) < 30:
                    continue

                lo, hi = clip_bounds(series)

                train_size = int(len(series) * 0.8)
                train, test = series[:train_size], series[train_size:]

                model = ARIMA(train, order=(1, 1, 1)).fit()
                preds = model.forecast(steps=len(test))
                rmse = round(np.sqrt(mean_squared_error(test, preds)), 4)

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

            except Exception as e:
                st.warning(f"Skipped {well} due to error: {e}")
                continue

        st.markdown("### 📈 ARIMA Model Metrics (All Wells)")
        st.dataframe(pd.DataFrame(arima_metrics), use_container_width=True)

        st.markdown("### 📅 ARIMA Forecast: Avg Depth per Year (2025–2029)")
        st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True)
