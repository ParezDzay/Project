import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# File paths - can be parameterized if needed
gw_file_path = GW data.csv"
output_path = GW data.csv"

def groundwater_data_page():
    st.title("Groundwater Data Over 20 Years")

    if not os.path.exists(output_path):
        st.error("Groundwater CSV file (missing filled) not found.")
        st.stop()

    # Load groundwater data with missing filled
    gw_df = pd.read_csv(output_path)

    try:
        # Prepare Date column for time series analysis
        gw_df["Year"] = gw_df["Year"].astype(int)
        gw_df["Months"] = gw_df["Months"].astype(int)
        gw_df["Date"] = pd.to_datetime(
            gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01",
            format="%Y-%m-%d"
        )

        # Define tabs for data exploration and cleaning workflow
        tab1, tab2, tab3 = st.tabs([
            "📉 Data with Missing",
            "✏️ Edit Raw Data",
            "✅ Data without Missing",
        ])

        # List of well columns excluding metadata columns
        well_cols = [col for col in gw_df.columns if col not in ["Year", "Months", "Date"]]

        # Tab 1: Display raw groundwater data (with missing values)
        with tab1:
            st.subheader("Raw Groundwater Table (with missing)")
            st.dataframe(gw_df, use_container_width=True)

        # Tab 2: Allow user to edit and save raw groundwater data
        with tab2:
            st.subheader("Edit and Save Raw Groundwater Data")
            edited_df = st.data_editor(gw_df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Edited Data"):
                edited_df.to_csv(gw_file_path, index=False)
                st.success("Groundwater data saved successfully.")

        # Tab 3: Show groundwater data without missing (filled data)
        with tab3:
            st.subheader("Groundwater Table (Missing Data Filled)")
            st.dataframe(gw_df, use_container_width=True)

            st.subheader("📊 Groundwater Trends (Depth Plot)")
            selected_wells = st.multiselect("Select wells to display:", well_cols, default=well_cols[:3])

            if selected_wells:
                melted = gw_df.melt(
                    id_vars=["Date"], 
                    value_vars=selected_wells, 
                    var_name="Well", 
                    value_name="GW_Level"
                )
                melted["GW_Level"] = -melted["GW_Level"]  # Negative depth for plotting
                
                fig = px.line(
                    melted,
                    x="Date",
                    y="GW_Level",
                    color="Well",
                    title="Monthly Groundwater Depth Over Time",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one well.")

    except Exception as e:
        st.error(f"Error processing groundwater data: {e}")
