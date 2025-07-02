import streamlit as st
import pandas as pd
import numpy as np
import os
import pymannkendall as mk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime  # for a load-timestamp banner (optional)

def groundwater_trends_page():
    output_path = "GW data (missing filled).csv"

    # — optional: see immediately whether the module reloaded —
    st.caption(f"⏱️ trend.py last loaded: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    st.title("📉 Groundwater Trends for Wells (MK, Sen’s Slope, MMK)")

    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    df = pd.read_csv(output_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    well_columns = [col for col in df.columns if col.startswith("W")]

    # Tabs
    tab_mk, tab_ita, tab_ita_plot = st.tabs(
        ["📊 MK, Sen’s Slope & MMK", "💡 ITA Analysis", "📈 ITA Plot"]
    )

    # ------------------------------------------------------------------
    # Helper: map MK/MMK outcome to "Increasing / Decreasing / No Trend"
    # ------------------------------------------------------------------
    def trend_label(p_value, tau):
        """
        For depth-to-water:
            +τ (depth rises)  -> "Decreasing" groundwater
            –τ (depth falls)  -> "Increasing" groundwater
        """
        if p_value < 0.05:
            return "Decreasing" if tau > 0 else "Increasing"
        return "No Trend"

    # === MK Tab =======================================================
    with tab_mk:
        st.subheader("Mann-Kendall, Sen’s Slope, and Modified MK Analysis")
        annual_rows = []
        for well in well_columns:
            yearly = df.groupby("Year")[well].mean().dropna()
            if len(yearly) <= 10:
                continue

            mk_out  = mk.original_test(yearly)
            mmk_out = mk.hamed_rao_modification_test(yearly)
            label   = trend_label(mmk_out.p, mmk_out.Tau)

            annual_rows.append(
                [
                    well,
                    round(mk_out.Tau, 3),   round(mk_out.z, 3),   round(mk_out.p, 4),
                    round(mk_out.slope, 3),
                    round(mmk_out.Tau, 3),  round(mmk_out.z, 3),  round(mmk_out.p, 4),
                    label,
                ]
            )

        multi_cols = pd.MultiIndex.from_tuples(
            [
                ("Well", ""),
                ("MK", "Tau"), ("MK", "Z-Statistic"), ("MK", "P-Value"),
                ("Sen’s Slope", "Slope"),
                ("MMK", "Tau"), ("MMK", "Z-Statistic"), ("MMK", "P-Value"),
                ("MMK", "Trend"),
            ]
        )
        st.dataframe(pd.DataFrame(annual_rows, columns=multi_cols),
                     use_container_width=True)

    # === ITA Tab ======================================================
    with tab_ita:
        st.subheader("ITA Analysis – Trend Metrics")
        ita_rows = []

        for well in well_columns:
            yearly = df.groupby("Year")[well].mean().dropna()
            if len(yearly) < 2:
                continue

            x = np.arange(len(yearly))
            y = yearly.values
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x + intercept
            r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)
            sigma = np.std(y)
            sand, scrit = 0.5 * sigma, 0.95 * sigma

            if abs(slope) > scrit:
                ita_flag = "Significant Trend"
            elif abs(slope) > sand:
                ita_flag = "Possible Trend"
            else:
                ita_flag = "No Trend"

            hydro_dir = "Decreasing" if slope > 0 else ("Increasing" if slope < 0 else "Stable")
            ita_rows.append(
                {
                    "Well": well,
                    "Slope": round(slope, 4),
                    "Mean": round(np.mean(y), 3),
                    "Std Dev": round(sigma, 3),
                    "S": round(sand, 3),
                    "Scrit": round(scrit, 3),
                    "R²": round(r2, 4),
                    "Trend (ITA + Hydrological)": f"{ita_flag} ({hydro_dir})",
                }
            )

        st.dataframe(pd.DataFrame(ita_rows), use_container_width=True)

    # === ITA Plot Tab =================================================
    with tab_ita_plot:
        st.subheader("ITA Groundwater Level Comparison Per Well")
        annual_means = df.groupby("Year")[well_columns].mean().dropna()

        early_yrs  = list(range(2004, 2015))
        recent_yrs = list(range(2015, 2025))

        for well in well_columns:
            early = annual_means.loc[annual_means.index.isin(early_yrs),  well].dropna()
            late  = annual_means.loc[annual_means.index.isin(recent_yrs),  well].dropna()
            n = min(len(early), len(late))
            if n < 2:
                st.write(f"Not enough data to plot for well {well}.")
                continue

            x, y = early.values[:n], late.values[:n]

            reg = LinearRegression().fit(x.reshape(-1, 1), y)
            y_fit = reg.predict(x.reshape(-1, 1))
            r2 = reg.score(x.reshape(-1, 1), y)

            vmin = min(x.min(), y.min()) * 0.95
            vmax = max(x.max(), y.max()) * 1.05

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor("#FAF3E0")

            for xi, yi in zip(x, y):
                if yi > xi:   # depth deeper in 2015-24 → groundwater decline
                    ax.scatter(xi, yi, marker="▲", color="orange", s=80,
                               label="Decreasing" if "Decreasing" not in ax.get_legend_handles_labels()[1] else "")
                else:         # shallower depth → recovery
                    ax.scatter(xi, yi, marker="▼", color="green", s=80,
                               label="Increasing" if "Increasing" not in ax.get_legend_handles_labels()[1] else "")

            ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, label="1:1 Line")
            ax.plot(x, y_fit, color="blue", lw=2, label=f"Trend (R²={r2:.3f})")
            ax.set_xlabel("2004–2014")
            ax.set_ylabel("2015–2024")
            ax.set_title(f"ITA Plot – {well}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            st.pyplot(fig)
