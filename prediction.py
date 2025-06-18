import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def groundwater_prediction_page(data_path):
    st.title("📊 Groundwater Prediction (2025–2029)")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    wells = [col for col in df.columns if col.startswith("W")]
    meteo = ["Precipitation", "Temperature", "Humidity", "Evaporation"]
    forecast_years = [2025, 2026, 2027, 2028, 2029]
    H = len(forecast_years)

    tab_ann, tab_arima, tab_arimax = st.tabs(["🤖 ANN", "📉 ARIMA", "📊 ARIMAX"])

    # ------------------------- ANN -------------------------
    with tab_ann:
        st.subheader("ANN Forecast (excluding meteorological variables)")
        results = []
        for well in wells:
            s = df[well].dropna()
            if len(s) < 60:
                continue
            # Create lag features
            lag = 12
            data = pd.concat([s.shift(i) for i in range(lag, 0, -1)], axis=1)
            data.columns = [f"lag_{i}" for i in range(lag, 0, -1)]
            data["target"] = s.values
            data.dropna(inplace=True)

            train = data.iloc[:-H]
            test = data.iloc[-H:]

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(train.drop("target", axis=1))
            y_train = train["target"].values
            X_test = scaler.transform(test.drop("target", axis=1))
            y_test = test["target"].values

            model = Sequential([
                Dense(64, activation='relu', input_shape=(lag,)),
                Dense(20, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, verbose=0,
                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

            pred = model.predict(X_test).flatten()

            r2_train = r2_score(y_train, model.predict(X_train))
            rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
            r2_test = r2_score(y_test, pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, pred))

            row = {
                "Well": well,
                "R²_train": round(r2_train, 4),
                "RMSE_train": round(rmse_train, 4),
                "R²_test": round(r2_test, 4),
                "RMSE_test": round(rmse_test, 4)
            }
            for i, year in enumerate(forecast_years):
                row[str(year)] = round(pred[i], 2) if i < len(pred) else np.nan
            results.append(row)

        st.dataframe(pd.DataFrame(results), use_container_width=True)

    # ------------------------- ARIMA -------------------------
    with tab_arima:
        st.subheader("ARIMA Forecast (excluding meteorological variables)")
        results = []
        for well in wells:
            s = df[well].dropna()
            try:
                model = ARIMA(s, order=(1, 1, 1)).fit()
                pred = model.forecast(steps=H)
                future_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=H, freq="MS")
                annual = pd.Series(pred.values, index=future_index).resample("Y").mean()
                row = {
                    "Well": well,
                    "R²_train": round(model.rsquared if hasattr(model, "rsquared") else np.nan, 4),
                    "RMSE_train": round(np.sqrt(model.mse), 4) if hasattr(model, "mse") else np.nan
                }
                for year in forecast_years:
                    val = annual[annual.index.year == year]
                    row[str(year)] = round(val.iloc[0], 2) if not val.empty else np.nan
                results.append(row)
            except:
                continue
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    # ------------------------- ARIMAX -------------------------
    with tab_arimax:
        st.subheader("ARIMAX Forecast (including meteorological variables)")
        results = []
        for well in wells:
            s = df[well].dropna()
            endog = s
            exog = df[meteo].loc[s.index]
            try:
                model = SARIMAX(endog, exog=exog, order=(1, 1, 1)).fit(disp=False)
                future_exog = df[meteo].iloc[-H:]
                pred = model.forecast(steps=H, exog=future_exog)
                future_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=H, freq="MS")
                annual = pd.Series(pred.values, index=future_index).resample("Y").mean()
                row = {
                    "Well": well,
                    "R²_train": round(model.rsquared if hasattr(model, "rsquared") else np.nan, 4),
                    "RMSE_train": round(np.sqrt(model.mse), 4) if hasattr(model, "mse") else np.nan
                }
                for year in forecast_years:
                    val = annual[annual.index.year == year]
                    row[str(year)] = round(val.iloc[0], 2) if not val.empty else np.nan
                results.append(row)
            except:
                continue
        st.dataframe(pd.DataFrame(results), use_container_width=True)
