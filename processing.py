import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import statsmodels.api as sm

# File path
data_path = "GW data.csv"

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, rmse, mape

# --- Model functions ---
def run_ann(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test).flatten()
    return compute_metrics(y_test, y_pred)

def run_lstm(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test).flatten()
    return compute_metrics(y_test, y_pred)

def run_sarima(y_train, y_test):
    model = sm.tsa.SARIMAX(y_train, order=(0,1,1), seasonal_order=(0,1,1,12))
    results = model.fit(disp=False)
    y_pred = results.forecast(steps=len(y_test))
    return compute_metrics(y_test, y_pred)

# --- Data imputation functions ---
def remove_outliers(dataframe, well_cols):
    df_clean = dataframe.copy()
    for col in well_cols:
        series = df_clean[col]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean.loc[(series < lower) | (series > upper), col] = np.nan
    return df_clean

def calc_outlier_pct(df_in, well_cols):
    outlier_pct = {}
    for well in well_cols:
        series = df_in[well].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        pct = (len(outliers) / len(series)) * 100 if len(series) > 0 else 0
        outlier_pct[well] = round(pct, 2)
    return pd.Series(outlier_pct)

def apply_linear_interpolation(df_in):
    return df_in.interpolate(method='linear', limit_direction='both')

def apply_rf_imputer(df_in, well_cols):
    df_copy = df_in.copy()
    for col in well_cols:
        temp_df = df_copy.dropna(subset=[col])
        if temp_df.shape[0] < 10:
            continue
        features = [c for c in well_cols if c != col and c in temp_df.columns]
        if not features:
            continue
        X = temp_df[features]
        y = temp_df[col]
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X, y)
        null_rows = df_copy[col].isnull()
        X_pred = df_copy.loc[null_rows, features]
        if not X_pred.empty:
            try:
                df_copy.loc[null_rows, col] = rf.predict(X_pred)
            except Exception as e:
                st.warning(f"RF prediction skipped for {col}: {e}")
    return df_copy

def apply_knn_imputer(df_in, well_cols):
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(df_in[well_cols])
    df_knn = df_in.copy()
    df_knn[well_cols] = imputed_data
    return df_knn

def data_processing_page():
    st.title("Groundwater Data Processing")

    if not os.path.exists(data_path):
        st.error("Groundwater data file not found.")
        st.stop()

    df = pd.read_csv(data_path)
    try:
        df["Year"] = df["Year"].astype(int)
        df["Months"] = df["Months"].astype(int)
        df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Months"].astype(str) + "-01")
    except Exception as e:
        st.error(f"Date conversion error: {e}")
        return

    well_cols = [col for col in df.columns if col not in ["Year", "Months", "Date"]]

    tab1, tab2, tab3, tab_compare = st.tabs([
        "1️⃣ Linear Interpolation",
        "2️⃣ Random Forest",
        "3️⃣ KNN Imputation",
        "📊 Model Sensitivity Analysis"
    ])

    # --- Existing Imputation Tabs ---
    with tab1:
        st.header("Linear Interpolation")
        df_clean = remove_outliers(df, well_cols)
        df_linear = apply_linear_interpolation(df_clean)
        st.dataframe(df_linear, use_container_width=True)

    with tab2:
        st.header("Random Forest Imputation")
        df_clean = remove_outliers(df, well_cols)
        df_rf = apply_rf_imputer(df_clean, well_cols)
        st.dataframe(df_rf, use_container_width=True)

    with tab3:
        st.header("KNN Imputation")
        df_clean = remove_outliers(df, well_cols)
        df_knn = apply_knn_imputer(df_clean, well_cols)
        st.dataframe(df_knn, use_container_width=True)

    # --- New Sensitivity Analysis Tab ---
    with tab_compare:
        st.header("Model Sensitivity Analysis")

        imputation_methods = {
            "Linear Interpolation": apply_linear_interpolation,
            "Random Forest": apply_rf_imputer,
            "KNN": apply_knn_imputer
        }

        results = []

        for method_name, imputer_func in imputation_methods.items():
            st.write(f"Processing: {method_name}")
            df_clean = remove_outliers(df, well_cols)
            if method_name == "Random Forest" or method_name == "KNN":
                df_imputed = imputer_func(df_clean, well_cols)
            else:
                df_imputed = imputer_func(df_clean)

            for well in well_cols:
                y = df_imputed[well].values
                X = np.arange(len(y)).reshape(-1, 1)

                split_idx = int(len(y) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                ann_r2, ann_rmse, ann_mape = run_ann(X_train, y_train, X_test, y_test)
                lstm_r2, lstm_rmse, lstm_mape = run_lstm(X_train, y_train, X_test, y_test)
                sarima_r2, sarima_rmse, sarima_mape = run_sarima(y_train, y_test)

                results.append({
                    "Model": "ANN", "Imputation Method": method_name, "Well": well,
                    "R²": ann_r2, "RMSE": ann_rmse, "MAPE (%)": ann_mape
                })
                results.append({
                    "Model": "LSTM", "Imputation Method": method_name, "Well": well,
                    "R²": lstm_r2, "RMSE": lstm_rmse, "MAPE (%)": lstm_mape
                })
                results.append({
                    "Model": "SARIMA", "Imputation Method": method_name, "Well": well,
                    "R²": sarima_r2, "RMSE": sarima_rmse, "MAPE (%)": sarima_mape
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            label="Download Sensitivity Analysis Table",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="model_imputation_comparison.csv",
            mime="text/csv"
        )
