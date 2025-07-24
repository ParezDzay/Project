import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import plotly.express as px

HORIZON_M = 60  # 5 years

# ---------- Shared ----------

@st.cache_data(show_spinner=False)
def load_raw(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns:
        st.error(f"CSV must contain a 'Date' column. Found: {list(df.columns)}")
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

def plot_results(df_act, df_fit, df_for, well):
    forecast_start = df_act["Date"].max()
    plot_df = pd.concat([df_act, df_fit, df_for])

    fig = px.line(
        plot_df,
        x="Date",
        y="Depth",
        color="Type",
        line_dash="Type",
        labels={"Depth": "Water-table depth (m)", "Date": "Date", "Type": "Legend"},
        title=f"{well} — Model Fit & 5-Year Forecast",
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


# ---------- MLP ----------

def train_mlp(df_feat, well, layers, lags, scaler_type, lo, hi):
    X = df_feat.drop(columns=[well, "Date"])
    y = df_feat[well]

    n = len(df_feat)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    scaler = RobustScaler() if scaler_type == "Robust" else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    mdl = MLPRegressor(
        hidden_layer_sizes=layers,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=len(X_val) / (len(X_train) + len(X_val)),
        n_iter_no_change=20
    )
    mdl.fit(X_train_scaled, y_train)

    y_train_pred = np.clip(mdl.predict(X_train_scaled), lo, hi)
    y_val_pred = np.clip(mdl.predict(X_val_scaled), lo, hi)
    y_test_pred = np.clip(mdl.predict(X_test_scaled), lo, hi)

    df_feat.loc[X_train.index, "pred"] = y_train_pred
    df_feat.loc[X_val.index, "pred"] = y_val_pred
    df_feat.loc[X_test.index, "pred"] = y_test_pred

    metrics = {
        "R² train": round(r2_score(y_train, y_train_pred), 4),
        "RMSE train": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
        "R² test": round(r2_score(y_test, y_test_pred), 4),
        "RMSE test": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
    }

    r = df_feat.iloc[-1].copy()
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


# ---------- LSTM ----------

def train_lstm(df_feat, well, layers, lags, alpha, scaler_type, lo, hi):
    X = df_feat.drop(columns=[well, "Date"])
    y = df_feat[well]

    n = len(df_feat)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    scaler = RobustScaler() if scaler_type == "Robust" else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential()
    for i, units in enumerate(layers):
        return_sequences = i < len(layers) - 1
        if i == 0:
            model.add(LSTM(units, return_sequences=return_sequences, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        model.add(Dropout(alpha))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, epochs=300, batch_size=16,
              validation_data=(X_val_scaled, y_val),
              callbacks=[early_stop], verbose=0)

    y_train_pred = np.clip(model.predict(X_train_scaled).flatten(), lo, hi)
    y_val_pred = np.clip(model.predict(X_val_scaled).flatten(), lo, hi)
    y_test_pred = np.clip(model.predict(X_test_scaled).flatten(), lo, hi)

    df_feat.loc[X_train.index, "pred"] = y_train_pred
    df_feat.loc[X_val.index, "pred"] = y_val_pred
    df_feat.loc[X_test.index, "pred"] = y_test_pred

    metrics = {
        "R² train": round(r2_score(y_train, y_train_pred), 4),
        "RMSE train": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
        "R² test": round(r2_score(y_test, y_test_pred), 4),
        "RMSE test": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
    }

    return metrics, df_feat, pd.DataFrame()

# ---------- Streamlit ----------

raw = load_raw("GW_data_annual.csv")
if raw is None:
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]
tab1, tab2 = st.tabs(["🔮 MLP", "🔮 LSTM"])

with tab1:
    well = st.selectbox("Well", wells, key="mlp_well")
    clean = clean_series(raw, well)
    lo, hi = clip_bounds(clean[well])
    lags = st.slider("Lag steps", 1, 24, 12, key="mlp_lags")
    layers = tuple(int(x) for x in st.text_input("Hidden layers (comma)", "64,32", key="mlp_layers").split(",") if x.strip())
    scaler = st.selectbox("Scaler", ["Standard", "Robust"], key="mlp_scaler")
    feat = add_lags(clean, well, lags)
    metrics, hist, future = train_mlp(feat, well, layers, lags, scaler, lo, hi)
    st.json(metrics)
    plot_results(clean[["Date", well]].rename(columns={well: "Depth"}).assign(Type="Actual"),
                 hist[["Date", "pred"]].rename(columns={"pred": "Depth"}).assign(Type="Predicted"),
                 future.assign(Type="Forecast"), well)

with tab2:
    well = st.selectbox("Well", wells, key="lstm_well")
    clean = clean_series(raw, well)
    lo, hi = clip_bounds(clean[well])
    lags = st.slider("Lag steps", 1, 24, 12, key="lstm_lags")
    layers = tuple(int(x) for x in st.text_input("LSTM layers (comma)", "64,32", key="lstm_layers").split(",") if x.strip())
    alpha = st.slider("Dropout", 0.0, 0.5, 0.1, key="lstm_alpha")
    scaler = st.selectbox("Scaler", ["Standard", "Robust"], key="lstm_scaler")
    feat = add_lags(clean, well, lags)
    metrics, hist, _ = train_lstm(feat, well, layers, lags, alpha, scaler, lo, hi)
    st.json(metrics)
