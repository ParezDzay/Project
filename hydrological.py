import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from io import BytesIO
from zipfile import ZipFile
import matplotlib.patches as patches
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata
import pyproj
import contextily as ctx
import matplotlib.colors as mcolors

def hydrological_analysis_page():
    st.title("🌊 Hydrological Analysis")

    tab1, tab2, tab3 = st.tabs([
        "📉 Time Series Decomposition",
        "🌍 Groundwater Heatmap",
        "📍 Well Spatial Distribution"
    ])

    # === TAB 1: DECOMPOSITION VIEWER AND DOWNLOAD ===
    with tab1:
        file_path = r"C:\Parez\GW data (missing filled).csv"
        if not os.path.exists(file_path):
            st.error("Groundwater data file not found.")
            return

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        well_columns = [col for col in df.columns if col.startswith("W")]
        selected_well = st.selectbox("Select Well for Decomposition", well_columns, index=0)

        series = df[selected_well].resample("ME").mean().dropna()

        try:
            result = seasonal_decompose(series, model='additive', period=12)
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
                    plotted = 0
                    grouped_axes = []
                    labels = []

                    for j in range(2):
                        if i + j >= len(well_columns):
                            break
                        well = well_columns[i + j]
                        s = df[well].resample("ME").mean().dropna()
                        try:
                            r = seasonal_decompose(s, model='additive', period=12)
                        except:
                            continue

                        o = j * 4
                        r.observed.plot(ax=axes[o], color='black', linewidth=1.2)
                        axes[o].set_title("Original", fontsize=10)
                        axes[o].invert_yaxis()
                        r.trend.plot(ax=axes[o + 1], color='blue', linewidth=1.2)
                        axes[o + 1].set_title("Trend", fontsize=10)
                        axes[o + 1].invert_yaxis()
                        r.seasonal.plot(ax=axes[o + 2], color='green', linewidth=1.2)
                        axes[o + 2].set_title("Seasonal", fontsize=10)
                        r.resid.plot(ax=axes[o + 3], color='red', linewidth=1.2)
                        axes[o + 3].set_title("Residual", fontsize=10)
                        grouped_axes.append((axes[o], axes[o + 3]))
                        labels.append(well)
                        plotted += 1

                    for k in range(plotted * 4, 8):
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

    # === TAB 2: HEATMAP ===
    with tab2:
        st.subheader("🌍 Groundwater Trend Heatmap (m/year)")

        coord_path = r"C:\Parez\Wells detailed data.csv"
        if not os.path.exists(coord_path):
            st.error("Well coordinates file not found.")
            return

        coords_df = pd.read_csv(coord_path)
        coords_df.columns = coords_df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', '')
        coords_df = coords_df[coords_df['Well Name'].notna()].copy()

        df["Year"] = df.index.year
        well_columns = [col for col in df.columns if col.startswith("W")]

        slopes = []
        for well in well_columns:
            series = df.groupby("Year")[well].mean().dropna()
            if len(series) < 2:
                continue
            x = series.index.values.reshape(-1, 1)
            y = series.values
            model = LinearRegression().fit(x, y)
            slope = model.coef_[0]
            slopes.append((well, slope))

        trend_df = pd.DataFrame(slopes, columns=["W_Label", "Trend_m_per_year"])
        trend_df["Well Name"] = coords_df["Well Name"].values[:len(trend_df)]
        merged_df = pd.merge(trend_df, coords_df, on="Well Name", how="inner")

        transformer = pyproj.Transformer.from_crs("epsg:32638", "epsg:3857", always_xy=True)
        x = merged_df["GPS Coor. (UTM) X"].values
        y = merged_df["GPS Coor. (UTM) Y"].values
        mx, my = transformer.transform(x, y)

        merged_df["X"] = mx
        merged_df["Y"] = my

        grid_x, grid_y = np.meshgrid(
            np.linspace(min(mx), max(mx), 200),
            np.linspace(min(my), max(my), 200)
        )
        grid_z = griddata((mx, my), merged_df["Trend_m_per_year"], (grid_x, grid_y), method='cubic')

        bounds = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        colors = [
            "#313695", "#74add1", "#ffffbf", "#fee090", "#fdae61",
            "#f46d43", "#e34a33", "#d73027", "#bd0026", "#800026"
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_map", list(zip(np.linspace(0, 1, len(colors)), colors)))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=bounds, cmap=cmap, norm=norm, extend='both')

        ax.scatter(mx, my, c='black', s=30, label="Wells")
        for i, row in merged_df.iterrows():
            ax.text(row["X"], row["Y"], row["W_Label"], fontsize=8, ha='center', va='bottom')

        ctx.add_basemap(ax, crs="epsg:3857", source=ctx.providers.Esri.WorldImagery)
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='Slope (m/year)', shrink=0.8)

        ax.set_title("Groundwater Trend Heatmap (m/year)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        st.pyplot(fig)

    # === TAB 3: SPATIAL WELL DISTRIBUTION ===
    with tab3:
        st.subheader("📍 Spatial Distribution of Wells with Meteorological Data")

        gw_path = r"C:\Parez\GW_data_annual.csv"
        coord_path = r"C:\Parez\Wells detailed data.csv"
        if not os.path.exists(gw_path) or not os.path.exists(coord_path):
            st.error("Required data file(s) not found.")
            return

        df_annual = pd.read_csv(gw_path)
        coord_df = pd.read_csv(coord_path)
        coord_df.columns = coord_df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', '')
        coord_df = coord_df[coord_df['Well Name'].notna()].copy()
        coord_df["Well Name"] = coord_df["Well Name"].str.strip()

        well_cols = [col for col in df_annual.columns if col.startswith("W")]
        well_means = df_annual[well_cols].mean().reset_index()
        well_means.columns = ["Well Name", "Mean Value"]

        merged = pd.merge(well_means, coord_df, on="Well Name", how="inner")

        transformer = pyproj.Transformer.from_crs("epsg:32638", "epsg:3857", always_xy=True)
        merged["X"], merged["Y"] = transformer.transform(
            merged["GPS Coor. (UTM) X"].values,
            merged["GPS Coor. (UTM) Y"].values
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            merged["X"], merged["Y"],
            c=merged["Mean Value"],
            cmap="YlGnBu", s=120, edgecolor='black', alpha=0.85
        )

        for _, row in merged.iterrows():
            ax.text(row["X"], row["Y"] + 50, row["Well Name"], fontsize=8, ha='center', color='black')

        ctx.add_basemap(ax, crs="epsg:3857", source=ctx.providers.Esri.WorldImagery)
        plt.colorbar(sc, ax=ax, label="Mean Groundwater Level")
        ax.set_title("Spatial Distribution of Wells")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        plt.tight_layout()
        st.pyplot(fig)
