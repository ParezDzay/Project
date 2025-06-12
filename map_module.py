import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import pyproj
import numpy as np
from scipy.interpolate import griddata

def well_map_viewer_page(df: pd.DataFrame):
    st.title("Well Map Viewer")

    tab1, tab2 = st.tabs(["📊 Data Table", "🗺️ Map View"])

    # Ensure required columns are present
    required_cols = {"lat", "lon", "Depth (m)"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns: {required_cols - set(df.columns)}")
        return

    # Prepare data
    filtered_df = df.copy().reset_index(drop=True)
    filtered_df["Well Label"] = ["W" + str(i + 1) for i in range(len(filtered_df))]

    with tab1:
        st.subheader("Filtered Well Data")
        st.dataframe(filtered_df)

    with tab2:
        st.subheader("Well Depth Contour and Locations")

        # Transform to Web Mercator
        transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        mx, my = transformer.transform(filtered_df["lon"].values, filtered_df["lat"].values)

        filtered_df["x"] = mx
        filtered_df["y"] = my

        # Interpolate depth data
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(mx), max(mx), 200),
            np.linspace(min(my), max(my), 200)
        )
        grid_z = griddata(
            (filtered_df["x"], filtered_df["y"]),
            filtered_df["Depth (m)"],
            (grid_x, grid_y),
            method='cubic'
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=15, alpha=0.8)
        ax.scatter(filtered_df["x"], filtered_df["y"], c='red', edgecolor='black', s=80, zorder=3)

        for _, row in filtered_df.iterrows():
            ax.text(row["x"], row["y"] + 100, f"{row['Well Label']}\n{row['Depth (m)']} m",
                    fontsize=8, ha='center', va='bottom', color='white',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

        ctx.add_basemap(ax, crs="epsg:3857", source=ctx.providers.Esri.WorldImagery)

        cbar = fig.colorbar(contour, ax=ax, label="Well Depth (m)", shrink=0.8)
        ax.set_title("Well Depth Contour and Well Locations", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_xlim(min(filtered_df["x"]) - 2000, max(filtered_df["x"]) + 2000)
        ax.set_ylim(min(filtered_df["y"]) - 2000, max(filtered_df["y"]) + 2000)
        ax.grid(False)

        st.pyplot(fig)
