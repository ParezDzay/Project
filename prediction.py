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
from datetime import datetime
import plotly.express as px

# === Configuration ===
DATA_PATH = "GW_data_annual.csv"
HORIZON_Y = 5
FORECAST_YEARS = list(range(2025, 2030))

@st.cache_data(show_spinner=False)
def load_raw(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"], dayfirst=True)
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
    for _ in range(HORIZON_Y):
        for k in range(lags, 1, -1):
            r[f"{well}_lag{k}"] = r[f"{well}_lag{k - 1}"]
        r[f"{well}_lag1"] = r["pred"]
        nxt = r["Date"] + pd.DateOffset(years=1)
        r.update({"Date": nxt})
        val = np.clip(mdl.predict(scaler.transform(r[feats].to_frame().T))[0], lo, hi)
        r[well] = r["pred"] = val
        fut.append({"Year": nxt.year, "Depth": val})
    return metrics, df_feat, pd.DataFrame(fut)

def groundwater_prediction_page():
    st.title("📊 Groundwater Forecasting")

    raw = load_raw(DATA_PATH)
    if raw is None:
        st.error("CSV file not found.")
        return

    wells = [c for c in raw.columns if c.startswith("W")]
    exog_vars = [c for c in raw.columns if c not in wells + ["Date", "Year"]]

    model = st.radio("Choose Model", ["🔮 ANN", "📈 ARIMA", "📊 ARIMAX"], horizontal=True)

    if model == "🔮 ANN":
        well = st.sidebar.selectbox("Select Well", wells)
        clean = clean_series(raw, well)
        lo, hi = clip_bounds(clean[well])
        lags = st.sidebar.slider("Lag steps", 1, 10, 3)
        layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())
        scaler_choice = st.sidebar.selectbox("Scaler", ["Standard", "Robust"])
        feat = add_lags(clean, well, lags)
        metrics, hist, future = train_ann(feat, well, layers, lags, scaler_choice, lo, hi)

        st.subheader("ANN Model Metrics")
        st.json(metrics)

        df_act = pd.DataFrame({"Date": clean["Date"], "Depth": clean[well], "Type": "Actual"})
        df_fit = hist[["Date", "pred"]].rename(columns={"pred": "Depth"}).assign(Type="Predicted")
        df_for = future.assign(Type="Forecast", Date=pd.to_datetime(future["Year"], format="%Y"))
        plot_df = pd.concat([df_act, df_fit, df_for])

        fig = px.line(plot_df, x="Date", y="Depth", color="Type",
                      labels={"Depth": "Water-table depth (m)"},
                      title=f"{well} — ANN fit & 5-year forecast")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Table (2025–2029)")
        st.dataframe(df_for.style.format({"Depth": "{:.2f}"}), use_container_width=True)

    elif model == "📈 ARIMA":
        results = []
        for well in wells:
            try:
                s = clean_series(raw, well)
                s.index = raw["Date"]
                model = ARIMA(s, order=(1, 1, 1)).fit()
                pred = model.forecast(steps=HORIZON_Y)
                f = pd.Series(pred.values, index=pd.date_range(s.index[-1] + pd.DateOffset(years=1), periods=HORIZON_Y, freq='YS'))
                row = {"Well": well}
                for y in FORECAST_YEARS:
                    sel = f[f.index.year == y]
                    row[str(y)] = round(sel.iloc[0], 2) if not sel.empty else np.nan
                results.append(row)
            except Exception:
                continue
        st.subheader("ARIMA Forecast — 2025 to 2029")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    elif model == "📊 ARIMAX":
        results = []
        for well in wells:
            try:
                s = clean_series(raw, well)
                s.index = raw["Date"]
                exog = raw[exog_vars].fillna(method='ffill').fillna(method='bfill')
                exog.index = raw["Date"]
                exog = exog.loc[s.index]
                model = SARIMAX(s, exog=exog, order=(1, 1, 1),
                                enforce_stationarity=False, enforce_invertibility=False).fit()
                future_exog = exog[-HORIZON_Y:].reset_index(drop=True)
                pred = model.forecast(steps=HORIZON_Y, exog=future_exog)
                f = pd.Series(pred.values, index=pd.date_range(s.index[-1] + pd.DateOffset(years=1), periods=HORIZON_Y, freq='YS'))
                row = {"Well": well}
                for y in FORECAST_YEARS:
                    sel = f[f.index.year == y]
                    row[str(y)] = round(sel.iloc[0], 2) if not sel.empty else np.nan
                results.append(row)
            except Exception:
                continue
        st.subheader("ARIMAX Forecast — 2025 to 2029")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
