import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# File path
data_path = "GW data.csv"

# ---------------- Outlier Removal ----------------
def remove_outliers(df, well_cols):
    df_clean = df.copy()
    for col in well_cols:
        series = df_clean[col]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean.loc[(series < lower) | (series > upper), col] = np.nan
    return df_clean

def calc_outlier_pct(df, well_cols):
    outlier_pct = {}
    for well in well_cols:
        series = df[well].dropna()
        if len(series) == 0:
            outlier_pct[well] = 0
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        pct = (len(outliers) / len(series)) * 100
        outlier_pct[well] = round(pct, 2)
    return pd.Series(outlier_pct)

# ---------------- Imputation Methods ----------------
def apply_linear_interpolation(df):
    return df.interpolate(method="linear", limit_direction="both")

def apply_rf_imputer(df, well_cols):
    df_copy = df.copy()
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

def apply_knn_imputer(df, well_cols):
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(df[well_cols])
    df_knn = df.copy()
    df_knn[well_cols] = imputed_data
    return df_knn

# ---------------- Baseline Model ----------------
def baseline_model(X, y):
    try:
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).flatten()

        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(y) < 10:
            return np.nan, np.nan

        split_idx = int(len(y) * 0.8)
        if split_idx < 1 or split_idx >= len(y):
            return np.nan, np.nan

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(y_test) == 0 or len(y_train) == 0:
            return np.nan, np.nan

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return r2, rmse
    except Exception as e:
        st.warning(f"Baseline model error: {e}")
        return np.nan, np.nan

# ---------------- Streamlit App ----------------
def data_processing_page():
    st.title("Groundwater Data Processing — Imputation Method Decision")

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

    st.header("Step 1: Decide the Best Imputation Method")

    imputation_methods = {
        "Linear Interpolation": apply_linear_interpolation,
        "Random Forest": apply_rf_imputer,
        "KNN": apply_knn_imputer
    }

    decision_results = []

    for method_name, imputer_func in imputation_methods.items():
        st.write(f"Processing: {method_name}")

        df_clean = remove_outliers(df, well_cols)

        if method_name in ["Random Forest", "KNN"]:
            df_imputed = imputer_func(df_clean, well_cols)
        else:
            df_imputed = imputer_func(df_clean)

        outlier_pct = calc_outlier_pct(df_imputed, well_cols).mean()

        r2_list = []
        rmse_list = []

        for well in well_cols:
            y = df_imputed[well].values
            if np.isnan(y).all() or len(y) < 10:
                r2_list.append(np.nan)
                rmse_list.append(np.nan)
                continue

            X = np.arange(len(y))
            r2, rmse = baseline_model(X, y)
            r2_list.append(r2)
            rmse_list.append(rmse)

        avg_r2 = np.nanmean(r2_list)
        avg_rmse = np.nanmean(rmse_list)

        decision_results.append({
            "Imputation Method": method_name,
            "Average R²": avg_r2,
            "Average RMSE": avg_rmse,
            "Average Outlier %": outlier_pct
        })

    decision_df = pd.DataFrame(decision_results).sort_values(by="Average RMSE")
    st.dataframe(decision_df, use_container_width=True)

    st.download_button(
        label="Download Imputation Decision Table",
        data=decision_df.to_csv(index=False).encode("utf-8"),
        file_name="imputation_decision.csv",
        mime="text/csv"
    )

    st.write("✅ Choose the imputation method with the **lowest Average RMSE**, **highest Average R²**, and **lowest outlier %**.")

# ---------------- Run App ----------------
if __name__ == "__main__":
    data_processing_page()
