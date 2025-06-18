import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import plotly.express as px

HORIZON_M = 60
FORECAST_YEARS = list(range(2025, 2030))

@st.cache_data(show_spinner=False)
def load_raw(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if df.columns[0] != "Date":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, well):
    s = df[well].copy()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    s = s.where(s.between(q1 - 3 * iqr, q3 + 3 * iqr)).interpolate(limit_direction="both")
    return pd.DataFrame({"Date": df["Date"], well: s}).dropna().reset_index(drop=True)

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
        r.update({"Date": nxt})
        val = np.clip(mdl.predict(scaler.transform(r[feats].to_frame().T))[0], lo, hi)
        r[well] = r["pred"] = val
        fut.append({"Date": nxt, "Depth": val})
    return metrics, df_feat, pd.DataFrame(fut)

def groundwater_prediction_page(data_path="GW_data_annual.csv"):
    st.title("📊 Groundwater Forecasting")

    raw = load_raw(data_path)
    if raw is None:
        st.error("CSV file not found.")
        return

    wells = [c for c in raw.columns if c.startswith("W")]
    exog_vars = [c for c in raw.columns if c not in wells + ["Date"]]

    model = st.radio("Choose Model", ["🔮 ANN", "📈 ARIMA", "📊 ARIMAX"], horizontal=True)

    if model == "🔮 ANN":
        well = st.sidebar.selectbox("Select Well", wells)
        clean = clean_series(raw, well)
        lo, hi = clip_bounds(clean[well])
        lags = st.sidebar.slider("Lag steps", 1, 12, 6)
        layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())
        scaler_choice = st.sidebar.selectbox("Scaler", ["Standard", "Robust"])
        feat = add_lags(clean, well, lags)
        metrics, hist, future = train_ann(feat, well, layers, lags, scaler_choice, lo, hi)

        st.subheader("ANN Model Metrics")
        st.json(metrics)

        df_act = pd.DataFrame({"Date": clean["Date"], "Depth": clean[well], "Type": "Actual"})
        df_fit = hist[["Date", "pred"]].rename(columns={"pred": "Depth"}).assign(Type="Predicted")
        df_for = future.assign(Type="Forecast")
        plot_df = pd.concat([df_act, df_fit, df_for])

        fig = px.line(plot_df, x="Date", y="Depth", color="Type",
                      labels={"Depth": "Water-table depth (m)"},
                      title=f"{well} — ANN fit & 5-year forecast")
        fig.update_yaxes(autorange="reversed")
        for t in fig.data:
            if t.name == "Forecast":
                t.update(line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        df_for["Year"] = df_for["Date"].dt.year
        annual = df_for.groupby("Year")["Depth"].mean().reindex(FORECAST_YEARS).round(2).reset_index()
        st.subheader("Forecast Table (2025–2029)")
        st.dataframe(annual, use_container_width=True)

    elif model == "📈 ARIMA":
        results = []
        for well in wells:
            try:
                s = clean_series(raw, well)
                s.index = raw["Date"]
                s = s.asfreq("MS")
                if len(s.dropna()) < 24:
                    raise ValueError("Too few data points")

                model = ARIMA(s, order=(1, 1, 1)).fit()
                pred = model.forecast(steps=HORIZON_M)
                future_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=HORIZON_M, freq="MS")
                forecast = pd.Series(pred.values, index=future_index)
                annual = forecast.resample("Y").mean()

                row = {"Well": well}
                for y in FORECAST_YEARS:
                    val = annual[annual.index.year == y]
                    row[str(y)] = round(val.iloc[0], 2) if not val.empty else np.nan
                results.append(row)
            except Exception as e:
                st.info(f"⚠️ ARIMA failed for {well}: {e}")
                continue

        st.subheader("📈 ARIMA Forecast — Avg Annual Depth (2025–2029)")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("No ARIMA forecasts generated. Check your monthly data.")

    elif model == "📊 ARIMAX":
        results = []
        for well in wells:
            try:
                s = clean_series(raw, well)
                s.index = raw["Date"]
                s = s.asfreq("MS")
                if len(s.dropna()) < 24:
                    raise ValueError("Too few data points")

                exog = raw[exog_vars].fillna(method='ffill').fillna(method='bfill')
                exog.index = raw["Date"]
                exog = exog.loc[s.index]
                exog = exog.asfreq("MS")

                model = SARIMAX(s, exog=exog, order=(1, 1, 1),
                                enforce_stationarity=False, enforce_invertibility=False).fit()

                last_exog = exog.iloc[-1:].values
                future_exog = np.tile(last_exog, (HORIZON_M, 1))

                future_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=HORIZON_M, freq="MS")
                pred = model.forecast(steps=HORIZON_M, exog=future_exog)
                forecast = pd.Series(pred, index=future_index)
                annual = forecast.resample("Y").mean()

                row = {"Well": well}
                for y in FORECAST_YEARS:
                    val = annual[annual.index.year == y]
                    row[str(y)] = round(val.iloc[0], 2) if not val.empty else np.nan
                results.append(row)
            except Exception as e:
                st.info(f"⚠️ ARIMAX failed for {well}: {e}")
                continue

        st.subheader("📊 ARIMAX Forecast — Avg Annual Depth (2025–2029)")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("No ARIMAX forecasts generated. Check your monthly data.")
