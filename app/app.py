# Import Basic Dependencies
import streamlit as st
import pandas as pd
import numpy as npp
import joblib               # Extracted Machine Learning Models
from pathlib import Path

# Import Dependencies for Map Visualization
import folium
from streamlit_folium import st_folium

# Page Configuration (Home Tab's)
st.set_page_config(
    page_title="CalHouse Pricing",
)

# Load Data and Model
BASE_DIR = Path(__file__).resolve().parent.parent
st.write(BASE_DIR)

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR/"data"/"california_housing_train_fe.csv")

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR/"models"/"catboost_model1.pkl")

# Define Data and Model
df = load_data()
model = load_model()

# Title
st.image(BASE_DIR/"src"/"Photo-3-1800x1000.jpg")
st.title("Californian Housing Price Prediction")
st.markdown(
    """
    <div style="text-align: justify; max-width=800px; margin:auto">
    Discover the true value of any home in California instantly. 
    By leveraging the power of CatBoost machine learning, our app processes pinpoint map locations, property age, socio-economic income data, and your desired room configurations. 
    Whether you are buying or selling, get a personalized and highly reliable price prediction in seconds.
    This application predicts California housing prices
    using a CatBoost Regressor trained on the California
    Housing Dataset.
    Model Performance:
    - R² Score : 0.854
    - MAE      : 29,233
    - RMSE     : 44,245
    </div>
    """,
unsafe_allow_html=True)

# Sidebar Input
st.sidebar.header("Input Your Preferences!")

## House Age
house_age = st.sidebar.slider(
    "House Age (Years)",
    min_value = 1,
    max_value = 52,
    value = 1
)

## Annual Household Income
annual_income = st.sidebar.number_input(
    "Annual Household Income ($)",
    min_value = 5000,
    max_value = 150000,
    value = 5000,
    step=5000
)

## Rooms Number
total_rooms = st.sidebar.number_input(
    "Desired Room per-Households",
    min_value=1,
    max_value=15,
    value=1
)

## Bedroom Number
total_bedroom = st.sidebar.number_input(
    "Desired Bedroom per-Households",
    min_value=1,
    max_value=8,
    value=1
)

if total_bedroom > total_rooms:
    st.error(
        "Bedroom count cannot exceed room count."
    )
    st.stop()

# About Me
st.sidebar.markdown("---")
with st.sidebar.expander("About Me"):
    st.image(BASE_DIR/"src"/"Screenshot 2026-06-12 164759.png")
    st.write("""
             Albertus Antero Arnayusrandita

             Bachelor of Data Science (S.Si.D.)
             from Universitas Airlangga Surabaya

             Interested in:
             - Data Science
             - Data Analyst
             - Business Intelligence

             Email : randitaantero@gmail.com
             """)

# Inputation of Longitude and Latitude based on Real Maps
st.subheader("Select Your Preference Location!")
st.info("Click on the map to choose your desired location!")

## Map Access
m = folium.Map(
    location = [36.7783, -119.41799],
    zoom_start=6
)

map_data = st_folium(
    m,
    width=900,
    height=500
)

# Location Pin
selected_lat = None
selected_lon = None

if map_data["last_clicked"]:
    selected_lat = map_data["last_clicked"]["lat"]
    selected_lon = map_data["last_clicked"]["lng"]
    st.success(
        f"Selected Location: ({selected_lat:.4f}, {selected_lon:.4f})"
    )

# Prediction
if st.button("Predict House Price"):
    if selected_lat is None:
        st.error("Please select a location on the map first!")
    else:
        distance = (
            (df['latitude'] - selected_lat) ** 2
            +
            (df['longitude'] - selected_lon) ** 2
        )
        nearest_idx = distance.idxmin()
        nearest_area = df.loc[nearest_idx]
        population = nearest_area["population"]
        households = nearest_area["households"]
        area_income = nearest_area["median_income"]
        area_age = nearest_area["housing_median_age"]

        # Counting User Preference Input
        room_preference = total_rooms
        bedroom_preference = total_bedroom

        total_rooms = room_preference * households
        total_bedroom = bedroom_preference * households

        # Preproccessing : Convert income 
        median_income = annual_income / 10000
        # Preprocessing : Feature Engineering
        rooms_per_household = (total_rooms/households)
        bedrooms_per_room = (total_bedroom/total_rooms)
        population_per_household = (population/households)

        # Model Input
        input_df = pd.DataFrame({
            "longitude": [selected_lon],
            "latitude":[selected_lat],
            "housing_median_age":[house_age],
            "total_rooms":[total_rooms],
            "total_bedrooms":[total_bedroom],
            "population":[population],
            "households":[households],
            "median_income":[median_income],
            "rooms_per_household":[rooms_per_household],
            "bedrooms_per_room":[bedrooms_per_room],
            "population_per_household":[population_per_household]
        })
        prediction = model.predict(input_df)[0]

        # Results
        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Estimate House Price",
                f"${prediction:,.0f}"
            )
        with col2:
            st.metric(
                "Population",
                f"{int(population):,}"
            )
        with col3:
            st.metric(
                "Households",
                f"{int(households):,}"
            )
        
        # Result (Expand)
        st.subheader("Area's Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Area Median Household Income",
                f"${area_income * 10000:,.0f}",
            )
        with col2:
            st.metric(
                "Average House Age in Area",
                f"{area_age:.0f} years"
            )

# Footer
st.markdown("---")
st.caption(
    "Developed by Albertus Antero Arnayusrandita | California Housing Prediction Project"
)
