import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ====== Load Data ======
data_path = "GW data.csv"
df = pd.read_csv(data_path)

# ====== Preprocess ======
df["Year"] = df["Year"].astype(int)
df["Months"] = df["Months"].astype(int)
df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Months"].astype(str) + "-01")

well_cols = [col for col in df.columns if col not in ["Year", "Months", "Date"]]

# ====== Outlier Removal ======
def remove_outliers(df, wells):
    df_clean = df.copy()
    for col in wells:
        series = df_clean[col]
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_clean.loc[(series < lower) | (series > upper), col] = np.nan
    return df_clean

# ====== Imputation Methods ======
def linear_interpolation(df):
    return df.interpolate(method="linear", limit_direction="both")

def rf_imputer(df, wells):
    df_copy = df.copy()
    for col in wells:
        temp_df = df_copy.dropna(subset=[col])
        if temp_df.shape[0] < 5:
            continue
        features = [c for c in wells if c != col]
        if not features:
            continue
        X, y = temp_df[features], temp_df[col]
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X, y)
        mask = df_copy[col].isnull()
        try:
            df_copy.loc[mask, col] = rf.predict(df_copy.loc[mask, features])
        except:
            continue
    return df_copy

def knn_imputer(df, wells):
    imputer = KNNImputer(n_neighbors=3)
    df_copy = df.copy()
    df_copy[wells] = imputer.fit_transform(df_copy[wells])
    return df_copy

# ====== Baseline Model ======
def baseline_model(X, y):
    try:
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).flatten()
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if len(y) < 3:  # reduced threshold
            return np.nan, np.nan
        split_idx = max(int(len(y) * 0.8), 1)
        if split_idx >= len(y):
            split_idx = len(y) - 1
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        if len(y_test) == 0 or len(y_train) == 0:
            return np.nan, np.nan
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse
    except Exception as e:
        return np.nan, np.nan

# ====== Compare Methods ======
methods = {
    "Linear Interpolation": linear_interpolation,
    "Random Forest": rf_imputer,
    "KNN": knn_imputer
}

results = []
df_clean = remove_outliers(df, well_cols)

for method_name, func in methods.items():
    print(f"Running: {method_name}")
    if method_name == "Linear Interpolation":
        df_imputed = func(df_clean)
    else:
        df_imputed = func(df_clean, well_cols)

    # Calculate Outlier %
    df_outliers_removed = remove_outliers(df_imputed, well_cols)
    outlier_pct = df_outliers_removed[well_cols].isna().mean().mean() * 100

    r2_list, rmse_list = [], []
    for well in well_cols:
        y = df_imputed[well].values
        X = np.arange(len(y))
        r2, rmse = baseline_model(X, y)
        if not np.isnan(r2):
            r2_list.append(r2)
        if not np.isnan(rmse):
            rmse_list.append(rmse)

    avg_r2 = np.mean(r2_list) if r2_list else np.nan
    avg_rmse = np.mean(rmse_list) if rmse_list else np.nan

    results.append({
        "Method": method_name,
        "Average R²": avg_r2,
        "Average RMSE": avg_rmse,
        "Average Outlier %": outlier_pct
    })

# ====== Display Table ======
results_df = pd.DataFrame(results).sort_values(by="Average RMSE")
print("\nImputation Method Comparison:")
print(results_df)

results_df.to_csv("imputation_decision.csv", index=False)
print("\n✅ Saved results to 'imputation_decision.csv'")
