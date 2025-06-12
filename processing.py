import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

# File path
data_path = GW data.csv"

def data_processing_page():
    st.title("Groundwater Data Processing")

    if not os.path.exists(data_path):
        st.error("Groundwater data file not found.")
        st.stop()

    df = pd.read_csv(data_path)

    # Convert Year and Months to datetime Date column
    try:
        df["Year"] = df["Year"].astype(int)
        df["Months"] = df["Months"].astype(int)
        df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Months"].astype(str) + "-01")
    except Exception as e:
        st.error(f"Date conversion error: {e}")
        return

    well_cols = [col for col in df.columns if col not in ["Year", "Months", "Date"]]

    def remove_outliers(dataframe):
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

    def calc_outlier_pct(df_in):
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

    def apply_rf_imputer(df_in):
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

    def apply_knn_imputer(df_in):
        imputer = KNNImputer(n_neighbors=3)
        imputed_data = imputer.fit_transform(df_in[well_cols])
        df_knn = df_in.copy()
        df_knn[well_cols] = imputed_data
        return df_knn

    def convert_df_to_csv(df):
        # Cache the conversion to prevent recomputation
        return df.to_csv(index=False).encode('utf-8')

    # ----------------- Tab Interface -------------------
    tab1, tab2, tab3, tab_compare = st.tabs([
        "1️⃣ Linear Interpolation",
        "2️⃣ Random Forest",
        "3️⃣ KNN Imputation",
        " Compare Outliers"
    ])

    with tab1:
        st.header("Linear Interpolation")
        df_clean = remove_outliers(df)
        df_linear = apply_linear_interpolation(df_clean)
        st.dataframe(df_linear, use_container_width=True)

        csv_linear = convert_df_to_csv(df_linear)
        st.download_button(
            label="Download Linear Interpolated Data as CSV",
            data=csv_linear,
            file_name='linear_interpolated_data.csv',
            mime='text/csv',
        )

    with tab2:
        st.header("Random Forest Imputation")
        df_clean = remove_outliers(df)
        df_rf = apply_rf_imputer(df_clean)
        st.dataframe(df_rf, use_container_width=True)

        csv_rf = convert_df_to_csv(df_rf)
        st.download_button(
            label="Download Random Forest Imputed Data as CSV",
            data=csv_rf,
            file_name='rf_imputed_data.csv',
            mime='text/csv',
        )

    with tab3:
        st.header("KNN Imputation")
        df_clean = remove_outliers(df)
        df_knn = apply_knn_imputer(df_clean)
        st.dataframe(df_knn, use_container_width=True)

        csv_knn = convert_df_to_csv(df_knn)
        st.download_button(
            label="Download KNN Imputed Data as CSV",
            data=csv_knn,
            file_name='knn_imputed_data.csv',
            mime='text/csv',
        )

    with tab_compare:
        st.header("Compare Outlier % After Cleaning")

        df_clean = remove_outliers(df)
        dfs = {
            "Linear": apply_linear_interpolation(df_clean),
            "Random Forest": apply_rf_imputer(df_clean),
            "KNN": apply_knn_imputer(df_clean),
        }

        outlier_comparison = pd.DataFrame({
            method: calc_outlier_pct(df_meth)
            for method, df_meth in dfs.items()
        })

        st.dataframe(outlier_comparison.T.style.format("{:.2f}"), use_container_width=True)

        csv_compare = outlier_comparison.T.to_csv().encode('utf-8')
        st.download_button(
            label="Download Outlier Comparison as CSV",
            data=csv_compare,
            file_name='outlier_comparison.csv',
            mime='text/csv',
        )
