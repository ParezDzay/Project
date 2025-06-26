import streamlit as st 
import pandas as pd
import plotly.express as px
import os

from map_module import well_map_viewer_page
from data import groundwater_data_page
from trend import groundwater_trends_page
from processing import data_processing_page
from home import home_page
import hydrological
from prediction import groundwater_prediction_page
import result  # ✅ NEW IMPORT

output_path = "GW data (missing filled).csv"

st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home", 
        "🌍 Location and Map", 
        "📈 GW Data", 
        "🛠️ Data Processing",
        "🌊 Hydrological Analysis",
        "📉 Trends Analysis",
        "📊 GW Prediction",
        "📸 Result Visualization"  # ✅ NEW PAGE
    ]
)

file_path = "Wells detailed data.csv"

df = None
if page in ["🌍 Location and Map"]:
    if not os.path.exists(file_path):
        st.error("Well CSV file not found.")
        st.stop()
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
    df["Coordinate X"] = pd.to_numeric(df.get("Coordinate X"), errors="coerce")
    df["Coordinate Y"] = pd.to_numeric(df.get("Coordinate Y"), errors="coerce")
    df["Depth (m)"] = pd.to_numeric(df.get("Depth (m)"), errors="coerce")
    df.rename(columns={"Coordinate X": "lat", "Coordinate Y": "lon"}, inplace=True)
    df = df.dropna(subset=["lat", "lon"])

if page == "🏠 Home":
    home_page()

elif page == "🌍 Location and Map":
    well_map_viewer_page(df)

elif page == "📈 GW Data":
    groundwater_data_page()

elif page == "📉 Trends Analysis":
    groundwater_trends_page()

elif page == "🛠️ Data Processing":
    data_processing_page()

elif page == "🌊 Hydrological Analysis":
    hydrological.hydrological_analysis_page()

elif page == "📊 GW Prediction":
    groundwater_prediction_page("GW_data_annual.csv")

elif page == "📸 Result Visualization":  # ✅ NEW PAGE CALL
    result.main()
