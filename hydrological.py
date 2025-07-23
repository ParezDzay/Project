import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL  # ✅ Use STL instead!
import os
from io import BytesIO
from zipfile import ZipFile
import matplotlib.patches as patches

def hydrological_analysis_page():
    st.title("🌊 Hydrological Analysis")

    tab1, tab2 = st.tabs([
        "📉 Time Series Decomposition",
        "🌍 Groundwater Heatmap",
    ])

    # === TAB 1: Decomposition ===
    with tab1:
        file_path = "GW data (missing filled).csv"
        if not os.path.exists(file_path):
            st.error("Groundwater data file not found.")
            return

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # ✅ Limit to only W1–W18
        well_columns = [col for col in df.columns if col.startswith("W")][:18]

        selected_well = st.selectbox("Select Well for Decomposition", well_columns, index=0)

        series = df[selected_well].resample("ME").mean().dropna()

        if len(series) < 12:
            st.warning(f"Not enough data for decomposition. Need at least 12 months, found {len(series)}.")
            return

        try:
            result = STL(series, period=12).fit()
        except Exception as e:
            st.error(f"Decomposition failed: {e}")
            return

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        result.observed.plot(ax=ax1, color='black')
        ax1.set_title(f"{selected_well} - Original")
        ax1.invert_yaxis()
        result.trend.plot(ax=ax2, color='blue')
        ax2.set_title("Trend (Moving Average)")
        ax2.invert_yaxis()
        result.seasonal.plot(ax=ax3, color='green')
        ax3.set_title("Seasonal Pattern (Monthly)")
        result.resid.plot(ax=ax4, color='red')
        ax4.set_title("Residual (Noise/Anomalies)")
        plt.tight_layout()
        st.pyplot(fig)

        if st.button("📥 Download All Decompositions (2 Wells per A4 JPG)"):
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                for i in range(0, len(well_columns), 2):
                    fig, axes = plt.subplots(8, 1, figsize=(8.27, 11.69), sharex=True)
                    fig.subplots_adjust(hspace=0.6)
                    grouped_axes = []
                    labels = []

                    for j in range(2):
                        if i + j >= len(well_columns):
                            break
                        well = well_columns[i + j]
                        s = df[well].resample("ME").mean().dropna()
                        if len(s) < 12:
                            continue
                        try:
                            r = STL(s, period=12).fit()
                        except:
                            continue

                        o = j * 4
                        r.observed.plot(ax=axes[o], color='black')
                        axes[o].set_title("Original", fontsize=10)
                        axes[o].invert_yaxis()
                        r.trend.plot(ax=axes[o + 1], color='blue')
                        axes[o + 1].set_title("Trend", fontsize=10)
                        axes[o + 1].invert_yaxis()
                        r.seasonal.plot(ax=axes[o + 2], color='green')
                        axes[o + 2].set_title("Seasonal", fontsize=10)
                        r.resid.plot(ax=axes[o + 3], color='red')
                        axes[o + 3].set_title("Residual", fontsize=10)
                        grouped_axes.append((axes[o], axes[o + 3]))
                        labels.append(well)

                    for k in range(len(labels) * 4, 8):
                        fig.delaxes(axes[k])

                    for idx, (top_ax, bottom_ax) in enumerate(grouped_axes):
                        fig_x0, fig_y1 = top_ax.get_position().x0, top_ax.get_position().y1
                        fig_x1, fig_y0 = top_ax.get_position().x1, bottom_ax.get_position().y0
                        rect = patches.FancyBboxPatch(
                            (fig_x0, fig_y0),
                            fig_x1 - fig_x0,
                            fig_y1 - fig_y0 + 0.01,
                            boxstyle="round,pad=0.01",
                            edgecolor="gray",
                            facecolor="whitesmoke",
                            linewidth=1.2,
                            transform=fig.transFigure,
                            zorder=-1
                        )
                        fig.patches.append(rect)
                        fig.text(
                            x=fig_x0 + 0.01,
                            y=fig_y1 + 0.005,
                            s=f"{labels[idx]}",
                            fontsize=11,
                            fontweight="bold",
                            color="darkblue",
                            transform=fig.transFigure
                        )

                    fig.suptitle(f"Decomposition: {labels[0]} to {labels[1] if len(labels) > 1 else ''}",
                                 fontsize=12, y=0.96)

                    buf = BytesIO()
                    fig.savefig(buf, format='jpg', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    zip_file.writestr(f"Decomposition_{i//2 + 1}.jpg", buf.read())
                    plt.close(fig)

            zip_buffer.seek(0)
            st.download_button(
                label="📄 Download 2-Well Decomposition JPGs (Zipped)",
                data=zip_buffer,
                file_name="Well_Decompositions_2WellsPerJPG.zip",
                mime="application/zip"
            )
