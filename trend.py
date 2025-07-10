import streamlit as st
import pandas as pd
import numpy as np
import os
import pymannkendall as mk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def groundwater_trends_page():
    output_path = "GW data (missing filled).csv"

    st.title("📉 Groundwater Trends for Wells (MK, Sen’s Slope, MMK)")

    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    df = pd.read_csv(output_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    well_columns = [col for col in df.columns if col.startswith('W')]

    # Tabs
    tab_mk, tab_ita, tab_ita_plot = st.tabs([
        "📊 MK, Sen’s Slope & MMK",
        "💡 ITA Analysis",
        "📈 ITA Plot"
    ])

    def trend_label(p_value, tau):
        if p_value < 0.05:
            return "Decreasing" if tau > 0 else "Increasing"
        else:
            return "No Trend"

    # === MK Tab ===
    with tab_mk:
        st.subheader("Mann-Kendall, Sen’s Slope, and Modified MK Analysis")
        annual_data = []
        for well in well_columns:
            data = df.groupby("Year")[well].mean().dropna()
            if len(data) > 10:
                mk_result = mk.original_test(data)
                mmk_result = mk.hamed_rao_modification_test(data)
                trend = trend_label(mmk_result.p, mmk_result.Tau)

                annual_data.append([
                    well,
                    round(mk_result.Tau, 3), round(mk_result.z, 3), round(mk_result.p, 4),
                    round(mk_result.slope, 3),
                    round(mmk_result.Tau, 3), round(mmk_result.z, 3), round(mmk_result.p, 4),
                    trend
                ])

        multi_columns = pd.MultiIndex.from_tuples([
            ('Well', ''),
            ('MK', 'Tau'), ('MK', 'Z-Statistic'), ('MK', 'P-Value'),
            ('Sen’s Slope', 'Slope'),
            ('MMK', 'Tau'), ('MMK', 'Z-Statistic'), ('MMK', 'P-Value'),
            ('MMK', 'Trend')
        ])

        trend_df = pd.DataFrame(annual_data, columns=multi_columns)
        st.dataframe(trend_df, use_container_width=True)

    # === ITA Tab ===
    with tab_ita:
        st.subheader("ITA Analysis – Trend Metrics")
        ita_results = []

        for well in well_columns:
            series = df.groupby("Year")[well].mean().dropna()
            if len(series) < 2:
                continue
            x = np.arange(len(series))
            y = series.values
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            std_dev = np.std(y)
            sand = 0.5 * std_dev
            scrit = 0.95 * std_dev

            if abs(slope) > scrit:
                ita_trend = "Significant Trend"
            elif abs(slope) > sand:
                ita_trend = "Possible Trend"

            if slope > 0:
                hydro_trend = "Decreasing"
            elif slope < 0:
                hydro_trend = "Recovering"
            else:
                hydro_trend = "Stable"

            combined_trend = f"{ita_trend} ({hydro_trend})"

            ita_results.append({
                "Well": well,
                "Slope": round(slope, 4),
                "Mean": round(np.mean(y), 3),
                "Std Dev": round(std_dev, 3),
                "S": round(sand, 3),
                "Scrit": round(scrit, 3),
                "R²": round(r_squared, 4),
                "Trend (ITA + Hydrological)": combined_trend
            })

        ita_df = pd.DataFrame(ita_results)
        st.dataframe(ita_df, use_container_width=True)

    # === ITA Plot Tab ===
    with tab_ita_plot:
        st.subheader("ITA Groundwater Level Comparison Per Well")
        annual_means = df.groupby("Year")[well_columns].mean().dropna()

        first_years = list(range(2004, 2015))
        second_years = list(range(2015, 2025))

        for well in well_columns:
            first_vals = annual_means.loc[annual_means.index.isin(first_years), well].dropna()
            second_vals = annual_means.loc[annual_means.index.isin(second_years), well].dropna()
            n_points = min(len(first_vals), len(second_vals))
            x = first_vals.values[:n_points]
            y = second_vals.values[:n_points]

            if n_points < 2:
                st.write(f"Not enough data to plot for well {well}.")
                continue

            X_reshape = x.reshape(-1, 1)
            reg = LinearRegression().fit(X_reshape, y)
            y_pred = reg.predict(X_reshape)
            r2 = reg.score(X_reshape, y)

            min_val = min(np.min(x), np.min(y)) * 0.95
            max_val = max(np.max(x), np.max(y)) * 1.05

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor('#FAF3E0')

            for xi, yi in zip(x, y):
                if yi < xi:
                    ax.scatter(xi, yi, marker='v', color='green', s=80,
                               label='Increase' if 'Increase' not in ax.get_legend_handles_labels()[1] else "")
                else:
                    ax.scatter(xi, yi, marker='^', color='orange', s=80,
                               label='Decrease' if 'Decrease' not in ax.get_legend_handles_labels()[1] else "")

            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='1:1 Line')
            ax.plot(x, y_pred, color='blue', lw=2, label=f'Trend (R²={r2:.3f})')

            ax.set_xlabel('2004–2014')
            ax.set_ylabel('2015–2024')
            ax.set_title(f'ITA Plot - {well}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            st.pyplot(fig)
