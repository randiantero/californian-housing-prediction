# Import Dependencies
import streamlit as st
import pandas as pd
import joblib
from sklearn.prepocessing import StandardScaler
from sklearn.neigbors import NearestNeighbors

# Load Dataset
df = pd.read_csv("D:\ProjectDS\Californian_Housing_Pred\data\california_housing_train_fe.csv")

# Load Models
catboost_model = joblib.load("D:\ProjectDS\Californian_Housing_Pred\models\catboost_model1.pkl")

# Define Recommendation Features
recommendation_features = [
    "housing_median_age",
    "median_income",
    "total_rooms",
    "total_bedrooms"
]

# Train Nearest Neighbors
scaler = StandardScaler()
X_rec = scaler.fit_transform(df[recommendation_features])
nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn_model.fit(X_rec)

# Streamlit App
## Web Title
st.title("California Housing Price Prediction and Recommendation")

## User Input
housing_age = st.slider("Housing Age", 1, 52, 20)
median_income = st.number_input("Median Income", value = 5.0)
total_rooms = st.number_input("Total Rooms", value = 3000)
total_bedrooms = st.number_input("Total Bedrooms", value = 600)

## Recommendation Button
if st.button(
    "Find Recommended Area"
):
    user_data = scaler.transform([[
        housing_age,
        median_income,
        total_rooms,
        total_bedrooms
    ]])

    distances, indices = nn_model.kneighbors(
        user_data
    )

    recommendation = df.iloc[
        indices[0][0]
    ]

    ### Data Area
    longitude = recommendation["longitude"]
    latitude = recommendation['latitude']
    population = recommendation['population']
    households = recommendation['households']

    ###