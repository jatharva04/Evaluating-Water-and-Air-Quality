import os
import math
import json
import pickle
import requests
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import warnings

# Suppress the irritating XGBoost serialization warning as we are in development
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import spacy

from config import *

# --- WQI Models & Explainers ---
wqi_model = None
wqi_scaler = None
WQI_FEATURES = None
wqi_shap_explainer = None
wqi_lime_explainer = None
wqi_forecast_model = None
WQI_FORECAST_FEATURES = None
wqi_forecast_lime_explainer = None

# --- AQI Models & Explainers ---
final_model = None
lgb_model = None
lgb_features = None
lgb_horizons = None
shap_explainer = None
lime_explainer = None
aqi_model = None

# --- Chatbot Models & Knowledge Bases ---
ner_model = None
textcat_model = None
AIR_KB = None
WATER_KB = None

def load_all_models_and_data():
    """
    Loads ALL models, data, and explainers into global scope,
    but only on the first request that needs them.
    """
    # Tell Python we intend to modify all the global variables
    global wqi_model, wqi_scaler, WQI_FEATURES, wqi_shap_explainer, wqi_lime_explainer
    global wqi_forecast_model, WQI_FORECAST_FEATURES, wqi_forecast_lime_explainer
    global final_model, lgb_model, lgb_features, lgb_horizons, shap_explainer, lime_explainer, aqi_model
    global ner_model, textcat_model, AIR_KB, WATER_KB

    # ✅ Check if the models are already loaded. If one is None, load all of them.
    if final_model is None:
        print("--- Loading all models and data for the first time... ---")
        
        # --- 1. Load Chatbot Models & KBs ---
        print("Loading NER model...")
        ner_model = spacy.load("models/ner_model_final_final8")
        print("Loading Textcat model...")
        textcat_model = spacy.load("models/textcat_model_final_final7")

        print("Loading knowledge bases...")
        with open("data/kb_air.json", 'r', encoding="utf8") as f:
            AIR_KB = json.load(f)
        with open("data/kb_water.json", 'r', encoding="utf8") as f:
            WATER_KB = json.load(f)

        # --- 2. Load AQI Classifier & Explainers ---
        print("Loading AQI classifier...")
        MODEL_PATH = os.path.join("models", "random_forest_aqi.pkl")
        final_model = joblib.load(MODEL_PATH)
        aqi_model = joblib.load(MODEL_PATH)
        
        print("Creating AQI explainers...")
        shap_explainer = shap.TreeExplainer(final_model)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.random.rand(100, len(MODEL_FEATURES)),
            feature_names=MODEL_FEATURES,
            class_names=['AQI 1', 'AQI 2', 'AQI 3', 'AQI 4', 'AQI 5'],
            mode='classification'
        )
        
        print("Loading AQI forecast model...")
        LGB_BUNDLE_PATH = os.path.join("models", "lightgbm_multi.pkl")
        lgb_bundle = joblib.load(LGB_BUNDLE_PATH)
        lgb_model = lgb_bundle["model"]
        lgb_features = lgb_bundle["features"]
        lgb_horizons = lgb_bundle["horizons"]

        # --- 3. Load WQI Models & Explainers ---
        print("Loading WQI real-time model bundle...")
        with open("models/xgboost_model.pkl", "rb") as file:
            model_bundle = pickle.load(file)
        wqi_model = model_bundle["model"]
        wqi_scaler = model_bundle["scaler"]
        WQI_FEATURES = model_bundle["features"]
        
        print("Creating WQI explainers...")
        X_train_wqi_df = joblib.load(os.path.join("models", "X_train_wqi.pkl"))
        wqi_shap_explainer = shap.Explainer(wqi_model)
        wqi_lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=wqi_scaler.transform(X_train_wqi_df),
            feature_names=WQI_FEATURES,
            mode='regression'
        )

        # --- 4. Load WQI Forecast Model & Explainer ---
        print("Loading WQI forecast model bundle...")
        WQI_FORECAST_MODEL_PATH = os.path.join("models", "random_forest_wqi_forecast.pkl")
        WQI_FEATURES_PATH = os.path.join("models", "wqi_forecast_features.pkl")
        X_TRAIN_WQI_FORECAST_PATH = os.path.join("models", "X_train_wqi_forecast.pkl")

        wqi_forecast_model = joblib.load(WQI_FORECAST_MODEL_PATH)
        WQI_FORECAST_FEATURES = joblib.load(WQI_FEATURES_PATH)
        X_train_wqi_forecast_df = joblib.load(X_TRAIN_WQI_FORECAST_PATH)

        print("Creating WQI forecast explainer...")
        wqi_forecast_lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train_wqi_forecast_df),
            feature_names=WQI_FORECAST_FEATURES,
            class_names=['WQI_1m', 'WQI_2m', 'WQI_3m', 'WQI_4m', 'WQI_5m', 'WQI_6m'],
            mode='regression'
        )
        
        print("--- All models and data loaded successfully. ---")


def rf_predict_now(features: dict, dt_timestamp=None) -> int:
    """Uses the Random Forest model to predict the current AQI."""
    feats = add_time_features(features, dt_timestamp)
    X = pd.DataFrame([feats]).reindex(columns=RF_FEATURES, fill_value=0)
    return int(aqi_model.predict(X)[0])

def calc_indian_aqi(pollutants: dict) -> int:
    """Calculates the Indian AQI dynamically, adhering to CPCB rules and handling float values."""
    
    # 1. Enforce CPCB Minimum Data Requirements
    has_pm = "pm2_5" in pollutants or "pm10" in pollutants
    valid_keys = [k for k in ["pm2_5", "pm10", "no2", "o3", "so2", "co"] if k in pollutants]
    
    if len(valid_keys) < 3 or not has_pm:
        return 0 # Or raise a ValueError depending on your app's needs

    def get_sub_index(cp, breakpoints):
        for bp in breakpoints:
            # Using < for the upper bound to prevent float values falling through gaps
            if bp[0] <= cp < bp[1]:
                return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (cp - bp[0]) + bp[2]
        
        # If it exactly hits or exceeds the max value of the last breakpoint
        bp = breakpoints[-1]
        return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (cp - bp[0]) + bp[2]

    # Continuous Breakpoints: [Cp_low, Cp_high, I_low, I_high]
    bp_pm25 = [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 10000, 401, 500)]
    bp_pm10 = [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 10000, 401, 500)]
    bp_no2 = [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), (180, 280, 201, 300), (280, 400, 301, 400), (400, 10000, 401, 500)]
    bp_o3 = [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), (168, 208, 201, 300), (208, 748, 301, 400), (748, 10000, 401, 500)]
    bp_so2 = [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 10000, 401, 500)]
    bp_co = [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10, 101, 200), (10, 17, 201, 300), (17, 34, 301, 400), (34, 1000, 401, 500)]

    indices = []
    if "pm2_5" in pollutants: indices.append(get_sub_index(pollutants["pm2_5"], bp_pm25))
    if "pm10" in pollutants: indices.append(get_sub_index(pollutants["pm10"], bp_pm10))
    if "no2" in pollutants: indices.append(get_sub_index(pollutants["no2"], bp_no2))
    if "o3" in pollutants: indices.append(get_sub_index(pollutants["o3"], bp_o3))
    if "so2" in pollutants: indices.append(get_sub_index(pollutants["so2"], bp_so2))
    if "co" in pollutants: 
        co_mg = pollutants["co"] / 1000.0
        indices.append(get_sub_index(co_mg, bp_co))
        
    return int(max(indices))

def get_aqi_for_city(city_name, locations, aqi_model):
    """Orchestrates the process of getting a real-time AQI prediction for a city."""
    city_name_lower = city_name.lower()
    # Find the location details from your locations data
    location_info = next((loc for loc in locations if loc["full_name"].lower() == city_name_lower), None)
    if not location_info:
        return f"Sorry, '{city_name.capitalize()}' is not one of my supported locations."
    # Get real-time pollutant data
    pollutants, actual_aqi, dt_timestamp = get_realtime_pollutants(location_info["lat"], location_info["lon"])
    if pollutants is None:
        return f"Sorry, I couldn't fetch the live air quality data for {city_name.capitalize()} at the moment."

    # Prepare features and predict
    features_for_prediction = { "latitude": location_info["lat"], "longitude": location_info["lon"], **pollutants }
    predicted_aqi_index = rf_predict_now(features_for_prediction, dt_timestamp)

    # Fetch the exact Indian 0-500 AQI dynamically using pollutant concentrations
    aqi_500_scale = calc_indian_aqi(pollutants)

    # Return a dictionary of results
    return {
        "city": city_name,
        "aqi_1_to_5_scale": predicted_aqi_index,
        "actual_aqi": actual_aqi,
        "aqi_500_scale": aqi_500_scale,
        "category": map_dashboard_index_to_category(predicted_aqi_index)
    }

def map_aqi_to_category(aqi_value):
    """Maps a numerical 0-500+ AQI value to its category name."""
    aqi_value = int(aqi_value)
    if 0 <= aqi_value <= 50: return "Good"
    elif 51 <= aqi_value <= 100: return "Satisfactory"
    elif 101 <= aqi_value <= 200: return "Moderate"
    elif 201 <= aqi_value <= 300: return "Poor"
    elif 301 <= aqi_value <= 400: return "Very Poor"
    else: return "Severe"

def map_dashboard_index_to_category(index_value):
    """Maps the simple 1-5 dashboard index to its category name."""
    mapping = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return mapping.get(index_value, "Unknown")

def map_wqi_to_category(wqi_value):
    """Maps a numerical 0-100 WQI value to its category name."""
    wqi_value = int(wqi_value)
    if wqi_value >= 63: return "Good"
    elif wqi_value >= 50: return "Satisfactory"
    elif wqi_value >= 38: return "Poor"
    else: return "Very Poor"


def get_all_water_data(force_refresh=False):
    """Predicts the WQI for all water locations using the trained ML model and caches the results."""
    global CACHE_WATER

    # Check if we have valid, recent data in our cache
    if not force_refresh and CACHE_WATER["data"] and CACHE_WATER["timestamp"]:
        age = datetime.now() - CACHE_WATER["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            # print("Serving water data from cache...")
            return CACHE_WATER["data"]
    
    print("[WQI] Calculating new WQI real-time data from CSV...")
    results = []
    try:
        water_df = pd.read_csv("data/water_rep_data.csv")
        # parameter_columns = ["pH", "dissolvedoxygen", "bod", "cod", "nitrate", "FecalColiform"]
        for i, row in water_df.iterrows():
            # 1. Assemble the feature dictionary from the CSV and current time
            features = {
                "pH": float(row["pH"]),
                "dissolvedoxygen": float(row["dissolvedoxygen"]),
                "bod": float(row["bod"]),
                "cod": float(row["cod"]),
                "nitrate": float(row["nitrate"]),
                "FecalColiform": float(row["FecalColiform"]),
                "Year": datetime.now().year,
                "Month": datetime.now().month,
                "latitude": float(row["lat"]),
                "longitude": float(row["lon"])
            }
            
            # 2. Create and scale the DataFrame for prediction
            input_df = pd.DataFrame([features]).reindex(columns=WQI_FEATURES, fill_value=0)
            input_scaled = wqi_scaler.transform(input_df)
            predicted_wqi = float(wqi_model.predict(input_scaled)[0])
            classification = get_wqi_category_and_remark(predicted_wqi)
            
            # 5. Bundle the fast results
            results.append({
                "id": int(i),
                "full_name": str(row["location_name"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "wqi": float(predicted_wqi),
                "classification": classification["classification"],
                "remark": classification["remark"],
                "scaled_features": input_scaled.tolist(), # Convert to native Python list for caching 
                "parameters": {
                    "pH": float(row.get("pH")),
                    "dissolvedoxygen": float(row.get("dissolvedoxygen")),
                    "bod": float(row.get("bod")),
                    "cod": float(row.get("cod")),
                    "nitrate": float(row.get("nitrate")),
                    "FecalColiform": float(row.get("FecalColiform"))
                }
            })
    except Exception as e:
        print(f"!!! FAILED TO PROCESS water data !!!")
        print(f"ERROR: {e}")

    # Save the new results and the current time to the cache
    CACHE_WATER["data"] = results
    CACHE_WATER["timestamp"] = datetime.now()
    return results

def get_water_location_ml_details(location_id):
    """Dynamically calculates the heavy SHAP, LIME, and Recommendations on-demand for a single water location."""
    all_water = get_all_water_data()
    loc = next((p for p in all_water if str(p["id"]) == str(location_id)), None)
    
    if not loc:
        return None
        
    if "recommendation" in loc:
        return loc
        
    import numpy as np
    input_scaled = np.array(loc["scaled_features"])
    
    # Run heavy ML computations for this single water location
    shap_explanation = get_wqi_shap_explanation(input_scaled)
    lime_explanation = get_wqi_lime_explanation(input_scaled)
    recommendation = get_water_recommendations(loc["wqi"], shap_explanation)
    
    loc["shap_explanation"] = shap_explanation
    loc["lime_explanation"] = lime_explanation
    loc["recommendation"] = recommendation
    
    return loc

def get_wqi_category_and_remark(wqi_value):
    """Returns a dictionary with the WQI classification and pollution remark."""
    wqi_value = float(wqi_value)
    if wqi_value >= 63: return {"classification": "Good to Excellent", "remark": "Non Polluted"}
    elif wqi_value >= 50: return {"classification": "Medium to Good", "remark": "Non Polluted"}
    elif wqi_value >= 38: return {"classification": "Bad", "remark": "Polluted"}
    else: return {"classification": "Bad to Very Bad", "remark": "Heavily Polluted"}

def get_water_recommendations(wqi_score: float, shap_explanation: list) -> dict:
    """Generates advanced, multi-faceted water quality recommendations."""
    recommendations = { "general": {}, "smart": [], "contextual": [] }
    
    classification_info = get_wqi_category_and_remark(wqi_score)

    # 1. General Recommendations based on WQI classification
    recommendations["general"] = {
        "title": f"🚱 Classification: {classification_info['classification']}",
        "advice": f"The water is considered {classification_info['remark']}. It's advised to use a reliable purifier before drinking."
    }

    # 2. Smart Recommendations based on top SHAP features
    if wqi_score < 63 and shap_explanation: # Only give smart tips for non-good water
        parameter_advice = {
            "bod": "💡 High BOD is a key factor. This often indicates pollution from organic waste like sewage. Advanced filtration (RO) is recommended.",
            "FecalColiform": "💡 Fecal Coliform is a major concern. This indicates contamination from sewage. Water MUST be boiled or treated with a UV purifier.",
            "nitrate": "💡 High Nitrate levels were influential. This can come from fertilizer runoff and is a risk for infants.",
            "pH": "💡 The pH level was a significant factor. An unusual pH can affect taste and pipe safety."
        }
        top_features = [item['feature'] for item in shap_explanation[:3]]
        for feature, advice in parameter_advice.items():
            if feature in top_features:
                recommendations["smart"].append(advice)

    # 3. Contextual Recommendations (e.g., based on season)
    current_month = datetime.now().month
    if current_month in [6, 7, 8]: # Monsoon season
        recommendations["contextual"].append("🌧️ Seasonal Tip: During monsoon, heavy rains can increase runoff, washing more contaminants into water bodies. Be extra cautious with water purification.")

    return recommendations

def get_wqi_shap_explanation(scaled_features_df: pd.DataFrame) -> list:
    """Generates SHAP explanation for the regressor's prediction."""
    # For a regressor, shap_values is a simple 2D array
    shap_values = wqi_shap_explainer.shap_values(scaled_features_df)
    
    # We just need the first (and only) row of shap_values
    shap_values_for_prediction = shap_values[0]

    explanation = [{"feature": f, "impact": float(i)} for f, i in zip(WQI_FEATURES, shap_values_for_prediction)]
    explanation.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return explanation[:6]

def get_wqi_forecast_for_row(row, current_date):
    """
    Takes a single row of water data, builds the correct features,
    and returns the 6-month WQI forecast.
    """
    # This is the exact sanitization function from your training script
    def sanitize_name(name):
        return "".join(c if c.isalnum() else '_' for c in str(name))

    # 1. Base numerical and temporal features
    features = {
        'latitude': float(row["lat"]),
        'longitude': float(row["lon"]),
        'pH': float(row["pH"]),
        'dissolved_oxygen': float(row["dissolvedoxygen"]),
        'bod': float(row["bod"]),
        'cod': float(row["cod"]),
        'nitrate': float(row["nitrate"]),
        'FecalColiform': float(row.get("FecalColiform", 0.0)),
        'year': current_date.year,
        'month': current_date.month,
    }

    # 2. One-hot encode the Station Name
    station_name = row["location_name"]
    # Create the column name by sanitizing the original station name
    station_column_name = f"Station_Name_{sanitize_name(station_name)}"
    if station_column_name in WQI_FORECAST_FEATURES:
        features[station_column_name] = True

    # 3. One-hot encode the current Season
    month = current_date.month
    season = "Summer" if month in [3, 4, 5, 6] else \
             "Winter" if month in [11, 12, 1, 2] else "Monsoon"
    
    season_column_name = f"Season_{season}"
    if season_column_name in WQI_FORECAST_FEATURES:
        features[season_column_name] = True
    
    # 4. Create DataFrame, reindex, and predict
    input_df = pd.DataFrame([features]).reindex(columns=WQI_FORECAST_FEATURES, fill_value=False)
    preds = wqi_forecast_model.predict(input_df)[0]
    
    # 5. Format the output
    return {f"{h}_month": float(p) for h, p in zip(WQI_HORIZONS, preds)}

def get_all_water_forecasts(force_refresh=False):
    """Generates future WQI predictions for all water locations."""
    global CACHE_WATER_FORECAST
    # Check if we have valid, recent data in our cache
    if not force_refresh and CACHE_WATER_FORECAST["data"] and CACHE_WATER_FORECAST["timestamp"]:
        age = datetime.now() - CACHE_WATER_FORECAST["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            # print("Serving water forecast data from cache...")
            return CACHE_WATER_FORECAST["data"]

    print("[WQI] Generating new 6-month WQI forecasts...")
    results = []
    try:
        water_base_df = pd.read_csv("data/water_rep_data.csv") # Your CSV with base parameters
        for i, row in water_base_df.iterrows():
            current_date = datetime.now()
            # --- UPDATE: Call the new, all-in-one function ---
            forecast = get_wqi_forecast_for_row(row, current_date)
            results.append({
                "id": i,
                "full_name": row["location_name"],
                "forecast": forecast
            })
    except Exception as e:
        print(f"!!! FAILED TO PROCESS water forecast data !!! ERROR: {e}")
    CACHE_WATER_FORECAST["data"] = results
    CACHE_WATER_FORECAST["timestamp"] = datetime.now()
    return results

def build_wqi_plotly_series(forecast_data: dict):
    """Formats a single location's WQI forecast data as {x, y} for easy plotting."""
    if not forecast_data:
        return []
    
    now = datetime.now()
    series = []
    
    for h_str, wqi in sorted(forecast_data.items(), key=lambda x: int(x[0].replace("_month", ""))):
        months_ahead = int(h_str.replace("_month", ""))
        future_date = now + relativedelta(months=months_ahead)  # ✅ better than timedelta(days=30)
        series.append({
            "x": future_date.isoformat(),  # ISO 8601 for Chart.js time scale
            "y": round(float(wqi), 2)      # keep values rounded to 2 decimals
        })
    
    return series

def get_wqi_lime_explanation(scaled_features_df: pd.DataFrame) -> list:
    """Generates LIME explanation for the WQI regressor's prediction."""
    instance_to_explain = scaled_features_df[0] # LIME needs a 1D array
    
    explanation = wqi_lime_explainer.explain_instance(
        instance_to_explain,
        wqi_model.predict, # For regressors, just use .predict
        num_features=5
    )
    return explanation.as_list()


def add_time_features(d: dict, dt_timestamp=None) -> dict:
    """Adds time-based features to a dictionary."""
    out = d.copy()
    # Use explicit timezone to prevent server drift
    india_tz = ZoneInfo("Asia/Kolkata")
    
    if dt_timestamp is not None:
        now = datetime.fromtimestamp(dt_timestamp, tz=ZoneInfo("UTC")).astimezone(india_tz)
    else:
        now = datetime.now(india_tz)
        
    hour = out.get("hour", now.hour)
    out["hour"] = hour
    out["day"] = out.get("day", now.day) if dt_timestamp is None else now.day
    out["month"] = out.get("month", now.month) if dt_timestamp is None else now.month
    out["dayofweek"] = out.get("dayofweek", now.weekday()) if dt_timestamp is None else now.weekday()
    
    out["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    out["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    return out

def get_realtime_pollutants(lat: float, lon: float) -> tuple:
    """Fetches real, live air pollution data and exact AQI from the OpenWeather API."""
    api_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {'lat': lat, 'lon': lon, 'appid': API_KEY}

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data and 'list' in data and data['list']:
            components = data['list'][0]['components']
            actual_aqi = data['list'][0]['main']['aqi']
            dt_timestamp = data['list'][0].get('dt')
            return components, actual_aqi, dt_timestamp
        else:
            return None, None, None

    except requests.exceptions.RequestException as e:
        print(f"API Error fetching OpenWeather data for lat={lat}, lon={lon}: {e}")
        return None, None, None

def get_shap_explanation(features_df: pd.DataFrame) -> list:
    """Generates SHAP explanation for the classifier's prediction."""
    prediction = final_model.predict(features_df)[0]
    shap_values = shap_explainer.shap_values(features_df)
    class_index = int(prediction) - 1
    
    if 0 <= class_index < len(shap_values):
        shap_values_for_prediction = shap_values[class_index][0]
    else:
        shap_values_for_prediction = shap_values[0][0]

    explanation = [{"feature": f, "impact": float(i)} for f, i in zip(MODEL_FEATURES, shap_values_for_prediction) if abs(i) > 1e-4]
    explanation.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return explanation[:6]

def predict_fn_for_lime(numpy_array):
    """
    A wrapper function for LIME. It converts the NumPy array back to a
    DataFrame with feature names before making a prediction.
    """
    # Reshape if necessary, as LIME might pass a 1D array
    if numpy_array.ndim == 1:
        numpy_array = numpy_array.reshape(1, -1)
    
    # Convert to a DataFrame with the correct column names
    df = pd.DataFrame(numpy_array, columns=MODEL_FEATURES)
    
    # Return the prediction probabilities, which LIME expects
    return final_model.predict_proba(df)

def get_lime_explanation(features_df: pd.DataFrame) -> list:
    """Generates LIME explanation for the classifier's prediction."""
    instance_to_explain = features_df.iloc[0].values
    
    # Step 1: Tell LIME to generate explanations for ALL possible class indices (0, 1, 2, 3, 4)
    explanation = lime_explainer.explain_instance(
        instance_to_explain,
        predict_fn_for_lime,
        num_features=5,
        labels=range(len(final_model.classes_))
    )
    
    # Step 2: Get the model's actual prediction (e.g., the value 2)
    predicted_class = final_model.predict(features_df)[0]
    
    # Step 3: Find the index for that prediction. For classes [1,2,3,4,5], the index for class 2 is 1.
    try:
        # Get the list of classes the model knows: [1, 2, 3, 4, 5]
        class_list = final_model.classes_.tolist()
        # Find the position of our prediction in that list
        predicted_class_index = class_list.index(predicted_class)
    except ValueError:
        # As a fallback, just use the first class if something goes wrong
        predicted_class_index = 0
        
    # Step 4: Ask for the explanation using the correct index
    return explanation.as_list(label=predicted_class_index)

def get_recommendations(aqi_class: int, shap_explanation: list) -> dict:
    """Generates advanced, multi-faceted health recommendations."""
    
    recommendations = {
        "general": {},   # Will hold a single, detailed general recommendation
        "smart": [],     # Will hold a list of tips based on SHAP
        "contextual": [] # Will hold a list of tips based on time/season
    }

    # --- 1. Detailed General Recommendations for each AQI Class ---
    if aqi_class == 1:
        recommendations["general"] = {
            "title": "✅ Excellent Air Quality",
            "advice": "It's a perfect day for all outdoor activities. Enjoy the fresh air!"
        }
    elif aqi_class == 2:
        recommendations["general"] = {
            "title": "👌 Good Air Quality",
            "advice": "Air quality is acceptable. Unusually sensitive individuals should consider reducing intense or prolonged outdoor exertion."
        }
    elif aqi_class == 3:
        recommendations["general"] = {
            "title": "😷 Sensitive Groups May Be Affected",
            "advice": "Members of sensitive groups (children, elderly, people with respiratory issues) may experience health effects. The general public is less likely to be affected."
        }
    elif aqi_class == 4:
        recommendations["general"] = {
            "title": "❗ Unhealthy for Everyone",
            "advice": "Health alert: everyone may begin to experience health effects. Reduce heavy exertion and consider limiting time outdoors."
        }
    elif aqi_class == 5:
        recommendations["general"] = {
            "title": "🚨 Hazardous Air Quality",
            "advice": "This is a health emergency. Everyone should avoid all outdoor exertion. Stay indoors, keep windows closed, and run an air purifier if available."
        }

    # --- 2. "Smart" Recommendations based on the top 3 SHAP features ---
    if aqi_class >= 3 and shap_explanation:
        # Create a dictionary of advice for different pollutants
        pollutant_advice = {
            "pm2_5": "💡 Fine Particulate Matter (PM2.5) is a key factor. These tiny particles are harmful. Consider wearing an N95 mask if you must go outside.",
            "pm10": "💡 Coarse Particulate Matter (PM10) from dust and construction is high. Try to avoid dusty areas and keep windows closed.",
            "o3": "💡 Ground-level Ozone (O3) is a major contributor. Ozone is often highest on sunny afternoons. Schedule activities for the morning or evening.",
            "no2": "💡 Nitrogen Dioxide (NO2), mainly from traffic, is elevated. Avoid walking or exercising near busy roads, especially during rush hour.",
            "co": "💡 Carbon Monoxide (CO) levels are high. Ensure good ventilation and stay away from areas with heavy traffic congestion."
        }
        
        # Check the top 3 features from SHAP
        top_features = [item['feature'] for item in shap_explanation[:3]]
        
        for feature, advice in pollutant_advice.items():
            if feature in top_features:
                recommendations["smart"].append(advice)

        # Check if time of day is a major factor
        if 'hour_sin' in top_features or 'hour_cos' in top_features:
            recommendations["smart"].append("💡 The time of day is significantly impacting air quality. Pollution levels may be much higher or lower at different times.")

    # --- 3. Contextual Recommendation based on the season in Mumbai ---
    current_month = datetime.now().month
    if current_month in [9, 10, 11]: # Post-monsoon / early winter
        recommendations["contextual"].append("🍂 Seasonal Tip: As the monsoon season ends, changes in wind patterns can sometimes lead to stagnant air and higher pollution. Be mindful of air quality over the coming weeks.")

    return recommendations

def _process_location(loc):
    """Helper function to process a single location concurrently."""
    try:
        base_pollutants, actual_aqi, dt_timestamp = get_realtime_pollutants(loc["lat"], loc["lon"])
        
        if base_pollutants is None:
            return {
                "id": loc["id"], 
                "full_name": loc["full_name"], 
                "lat": loc["lat"], 
                "lon": loc["lon"],
                "aqi_now": 0,          # Zero so aqiColor logic drops it to gray "data unavailable" state
                "actual_aqi": "-",
                "aqi_500_scale": "-",
                "forecast": {}, 
                "pollutants": None,
                "timed_features": {"latitude": loc["lat"], "longitude": loc["lon"]}
            }
            
        feat_dict = {"latitude": loc["lat"], "longitude": loc["lon"], **base_pollutants}
        timed_feat = add_time_features(feat_dict, dt_timestamp)
        X_raw = pd.DataFrame([timed_feat]).reindex(columns=MODEL_FEATURES, fill_value=0)
        
        # Predict AQI with RF Model (Super Fast, < 5ms)
        aqi_now = int(final_model.predict(X_raw)[0])
        forecast = lgb_predict_future(timed_feat)
        
        return {
            "id": loc["id"], 
            "full_name": loc["full_name"], 
            "lat": loc["lat"], 
            "lon": loc["lon"],
            "aqi_now": aqi_now,          # Predicted by Model
            "actual_aqi": actual_aqi,    # Fetched directly from API
            "aqi_500_scale": calc_indian_aqi(base_pollutants), # Real True 0-500 scale
            "forecast": forecast, 
            "pollutants": base_pollutants,
            "timed_features": timed_feat  # Store for dynamic SHAP/LIME computation
        }
    except Exception as e:
        print(f"!!! FAILED TO PROCESS LOCATION: {loc.get('full_name', 'N/A')} !!! ERROR: {e}")
        return None

def get_location_ml_details(location_id):
    """Dynamically calculates the heavy SHAP, LIME, and Recommendations on-demand for a single location."""
    all_locations = predict_all_locations()
    loc = next((p for p in all_locations if str(p["id"]) == str(location_id)), None)
    
    if not loc:
        return None
        
    # Check if we've already computed it for this cache cycle
    if "recommendations" in loc:
        return loc
         
    # Rebuild the dataframe from the fast base fetch
    X_raw = pd.DataFrame([loc["timed_features"]]).reindex(columns=MODEL_FEATURES, fill_value=0)
    
    # 🧠 Here we run the Heavy ML code!
    shap_explanation = get_shap_explanation(X_raw)
    lime_explanation = get_lime_explanation(X_raw)
    recommendations = get_recommendations(loc["aqi_now"], shap_explanation)
    
    # Cache it onto the object so subsequent clicks in the same 30m window are instant
    loc["shap_explanation"] = shap_explanation
    loc["lime_explanation"] = lime_explanation
    loc["recommendations"] = recommendations
    
    return loc

def predict_all_locations(force_refresh=False):
    """Predicts current and future AQI for all hardcoded locations."""
    global CACHE

    # Check if we have valid, recent data in our cache
    if not force_refresh and CACHE["data"] and CACHE["timestamp"]:
        age = datetime.now() - CACHE["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            # print("Serving air quality data from cache...")
            return CACHE["data"]

    # If cache is empty or old, fetch new data from the API
    print("[AQI] Fetching new live data from OpenWeather API (Concurrent Mode)...")
    
    results = []
    # Maximum of 10 threads to avoid hammering the Free API tier too hard
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all location processing tasks
        future_to_loc = {executor.submit(_process_location, loc): loc for loc in LOCATIONS}
        for future in future_to_loc:
            res = future.result()
            if res is not None:
                results.append(res)
    
    # Sort results by ID to keep the order consistent
    results.sort(key=lambda x: x["id"])

    # Save the new results and the current time to the cache
    CACHE["data"] = results
    CACHE["timestamp"] = datetime.now()
    return results
    
def lgb_predict_future(features: dict) -> dict:
    X = pd.DataFrame([features]).reindex(columns=lgb_features, fill_value=0)
    preds = lgb_model.predict(X)[0]
    return {f"{h}h": int(p) for h, p in zip(lgb_horizons, preds)}


# Global caching initialization
import threading
def _start_background_warmer():
    """Background Daemon to continually update SHAP/LIME ML models to avoid 39 second wait times"""
    def warmer_loop():
        # First load is synchronous at startup, so we just sleep first!
        while True:
            # Sleep 25 minutes (cache expires in 30)
            time.sleep(25 * 60)
            try:
                print("\\n[🔥 DAEMON] Waking up to preemptively warm up the heavy SHAP/LIME ML Cache...")
                predict_all_locations(force_refresh=True)
                get_all_water_data(force_refresh=True)
                get_all_water_forecasts(force_refresh=True)
                print("[🔥 DAEMON] ML Cache successfully preemptively warmed! Going back to sleep.\\n")
            except Exception as e:
                print(f"[🔥 DAEMON] Error warming cache: {e}")

    thread = threading.Thread(target=warmer_loop, daemon=True)
    thread.start()

# Load models and Start Background Polling Engine!
load_all_models_and_data()
_start_background_warmer() # Ignite the daemon thread!
