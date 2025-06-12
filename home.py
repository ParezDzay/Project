import streamlit as st
import base64
import os

def set_bg_from_local(image_path):
    if not os.path.exists(image_path):
        st.warning("Background image not found.")
        return

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-repeat: no-repeat;
        background-position: bottom center;
        background-size: 700px auto;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }}
    .footer {{
        position: fixed;
        left: 0;
        bottom: 8px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        color: #999;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def home_page():
    set_bg_from_local("background.jpg")

    st.markdown(
        """
        <h1 style='text-align: center; font-weight: bold;'>
            Machine Learning-Based Analysis of Groundwater Levels in Erbil City
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        **Features**: This application is developed to support the research work of a Master's thesis project.  
        - Explore 20 well data  
        - Visualize well locations on a map  
        - Groundwater analysis for 20 wells in the Central Sub-Basin (CSB) of Erbil City  
    """)

    st.markdown('<div class="footer">Created by Parez Dizayee</div>', unsafe_allow_html=True)
