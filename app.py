import streamlit as st 
import pandas as pd
import plotly.express as px
import os

from map_module import well_map_viewer_page
from data import groundwater_data_page
from trend import groundwater_trends_page
from prediction import groundwater_prediction_page
from processing import data_processing_page
from home import home_page  # ⬅️ newly imported
import hydrological  # ⬅️ NEW: Import the hydrological module

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
        "🌊 Hydrological Analysis",  # ⬅️ NEW: Only added this line
        "📉 Trends Analysis",
        "📊 GW Prediction"
    ]
)

file_path = "Wells detailed data.csv"

df = None
if page not in ["📈 GW Data", "📉 Trends Analysis", "🛠️ Data Processing", "📊 GW Prediction"]:
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
    df_coords = pd.read_csv("Wells detailed data.csv")
    well_map_viewer_page(df_coords)


elif page == "📈 GW Data":
    groundwater_data_page()

elif page == "📉 Trends Analysis":
    groundwater_trends_page()

elif page == "🛠️ Data Processing":
    data_processing_page()

elif page == "📊 GW Prediction":
    groundwater_prediction_page(output_path)

elif page == "🌊 Hydrological Analysis":  # ⬅️ NEW: Calls the new page
    hydrological.hydrological_analysis_page()
