# (Everything remains the same above...)

# ──────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    raw    = load_levels_raw()
    coords = load_coords()

    # Guarantee full 2004–2029
    all_years = pd.DataFrame({"Year": list(range(2004, 2030))})
    levels    = (
        all_years
        .merge(raw, on="Year", how="left")
        .assign(
            Period=lambda df: np.where(df["Year"] >= 2025, "forecast", "observed")
        )
    )

    well_cols = [c for c in levels.columns if c.startswith("W")]
    if not well_cols:
        st.error("No W# columns found.")
        return

    # Sidebar
    years_str = levels["Year"].astype(str)
    yr_sel    = st.sidebar.selectbox("Year", years_str, index=len(years_str)-1)
    grid_res  = st.sidebar.slider("Grid resolution (px)", 100, 500, 300, 50)
    n_levels  = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
    make_gif  = st.sidebar.button("Generate GIF")

    # Single year
    year_int = int(yr_sel)
    row      = levels.loc[levels["Year"] == year_int, well_cols].iloc[0]
    period   = levels.loc[levels["Year"] == year_int, "Period"].iloc[0].capitalize()
    df_year  = row.rename_axis("well").reset_index(name="level")
    df_year["well"] = df_year["well"].apply(normalise_well)
    merged   = df_year.merge(coords, on="well", how="inner").dropna(subset=["level","lat","lon"])

    if merged.empty:
        st.warning(f"No data for {period} {yr_sel}.")
    else:
        title = f"{period.upper()} — {yr_sel}"
        st.subheader(f"{period.capitalize()} data — {yr_sel}")
        st.image(draw_frame(
            merged["lon"],
            merged["lat"],
            merged["level"],
            merged["well"],
            title,
            grid_res,
            n_levels,
        ))
        with st.expander("Raw data"):
            st.dataframe(merged.set_index("well"), use_container_width=True)

    # GIF
    if make_gif:
        with st.spinner("Building GIF…"):
            frames: list[Image.Image] = []
            for _, row in levels.iterrows():
                yr     = int(row["Year"])
                period = row["Period"].capitalize()
                df_f   = row[well_cols].rename_axis("well").reset_index(name="level")
                df_f["well"] = df_f["well"].apply(normalise_well)
                df_f = df_f.merge(coords, on="well", how="inner").dropna(subset=["level","lat","lon"])
                if df_f.empty:
                    continue
                label = f"{period.upper()} — {yr}"
                frames.append(draw_frame(
                    df_f["lon"],
                    df_f["lat"],
                    df_f["level"],
                    df_f["well"],
                    label,
                    grid_res,
                    n_levels,
                ))

            if not frames:
                st.error("No frames generated.")
                return

            buf = io.BytesIO()
            frames[0].save(
                buf, format="GIF", save_all=True,
                append_images=frames[1:], duration=1200, loop=0
            )
            buf.seek(0)

        st.subheader("Time-series animation")
        st.image(buf.getvalue())
        st.download_button(
            "Download GIF",
            data=buf.getvalue(),
            file_name="water_table_animation.gif",
            mime="image/gif",
        )

# ✅ Added entry point for app.py integration
def result_page():
    main()
