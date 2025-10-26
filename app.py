from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Step 0: Load saved models

lightgbm_model = joblib.load(r"D:\MLLabProject\tuned_lightgbm_regressor_model.pkl")
xgboost_model = joblib.load(r"D:\MLLabProject\tuned_xgboost_regressor_model.pkl")

app = FastAPI(title="California Housing Price Prediction")


# Step 2: Define input schema

class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity_1H_OCEAN: float
    ocean_proximity_INLAND: float
    ocean_proximity_ISLAND: float
    ocean_proximity_NEAR_BAY: float
    ocean_proximity_NEAR_OCEAN: float
    bedroom_ratio: float
    household_rooms: float


# Step 3: Prediction endpoint for LightGBM

@app.post("/predict/lightgbm")
def predict_lightgbm(features: HouseFeatures):
    # Convert input to array
    input_data = np.array([list(features.dict().values())])
    prediction = lightgbm_model.predict(input_data)[0]
    return {"predicted_median_house_value": float(prediction)}


# Step 4: Prediction endpoint for XGBoost

@app.post("/predict/xgboost")
def predict_xgboost(features: HouseFeatures):
    # Convert input to array
    input_data = np.array([list(features.dict().values())])
    prediction = xgboost_model.predict(input_data)[0]
    return {"predicted_median_house_value": float(prediction)}
