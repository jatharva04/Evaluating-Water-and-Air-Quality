import os
import math
import json
import pickle
import requests
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib

import folium
from folium import Map, CircleMarker, LayerControl
from folium.plugins import HeatMapWithTime

import shap
import lime
import lime.lime_tabular

import spacy
import re
import math
from thefuzz import process, fuzz

# =========================
# 0) Setup Flask App
# =========================
app = Flask(__name__)

# This dictionary will store our API results temporarily.
CACHE = { "data": None, "timestamp": None }
CACHE_WATER = { "data": None, "timestamp": None }
CACHE_WATER_FORECAST = {"data": None, "timestamp": None}

CACHE_DURATION_MINUTES = 15
API_KEY = "6a10ef7f599b4187e0cd396738b9fa41"

# ==================================
# 1) Initialize All Models as None
# ==================================
# This section runs instantly on every reload

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

WQI_HORIZONS = [1, 2, 3, 4, 5, 6] # In months
LOCATIONS_CSV = "Locations.csv"

# Required features for the models
MODEL_FEATURES = [
    "latitude", "longitude", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "dayofweek", "hour_sin", "hour_cos"
]

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
        ner_model = spacy.load("./ner_model_final_final8")
        print("Loading Textcat model...")
        textcat_model = spacy.load("./textcat_model_final_final7")

        print("Loading knowledge bases...")
        with open("kb_air.json", 'r', encoding="utf8") as f:
            AIR_KB = json.load(f)
        with open("kb_water.json", 'r', encoding="utf8") as f:
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

# ------------- CHATBOT imports -----------------

SUPPORTED_LOCATIONS = {"panvel", "vashi", "airoli", "nerul", "thane", "worli", "juhu", "versova", "badlapur", "ambernath", "powai"}
locations_df = pd.read_csv("locations_chatbot.csv")
LOCATIONS_CHATBOT = [
    {"id": i, "full_name": row["location_name"], "lat": float(row["latitude"]), "lon": float(row["longitude"])}
    for i, row in locations_df.iterrows()
]

LOCATION_ALIASES = {
    # Alias (lowercase) : Official Name from CSV
    "amb": "Ambernath",
    "ambernat": "Ambernath",
    "ambernaht": "Ambernath",
    "tna": "Thane",
    "mumbai": "Mumbai City",
    "bombay": "Mumbai City",
    "pnvl": "New Panvel",
    "panvel": "New Panvel",
    "seaface": "Worli Seaface",
    "jnpt": "Uran",
    "csmt": "Mumbai City",
    "cst": "Mumbai City"
}

POLLUTANT_MAP = {
    "ozone": "o3",
    "o3": "o3",
    "sulphur dioxide": "so2",
    "so2": "so2",
    "so 2": "so2",
    "carbon monoxide": "co",
    "carbonmonoxide": "co",
    "co": "co",
    "no2": "no2",
    "nitrogen dioxide": "no2",
    "no 2": "no2",
    "pm 2.5": "pm2_5",
    "pm_2.5": "pm2_5",
    "pm2.5": "pm2_5",
    "particulate matter 2.5": "pm2_5",
    "particulate matter 10": "pm10",
    "pm10": "pm10"
}

# Values represent the upper bound of the category (e.g., PM2.5 is 'Good' if value < 10).
POLLUTANT_THRESHOLDS = {
    "so2":   {"good": 20, "fair": 80, "moderate": 250, "poor": 350},
    "no2":   {"good": 40, "fair": 70, "moderate": 150, "poor": 200},
    "pm10":  {"good": 20, "fair": 50, "moderate": 100, "poor": 200},
    "pm2_5": {"good": 10, "fair": 25, "moderate": 50, "poor": 75},
    "o3":    {"good": 60, "fair": 100, "moderate": 140, "poor": 180},
    "co":    {"good": 4400, "fair": 9400, "moderate": 12400, "poor": 15400}
}

def find_best_pollutant_match(query, pollutant_map, score_cutoff=80):
    """
    Finds the best pollutant match from a dictionary's keys using fuzzy matching.
    Returns the canonical API code (e.g., "pm2_5") if a good match is found.
    """
    # Get the list of user-facing names to search against (the keys of the map)
    search_keys = list(pollutant_map.keys())
    # Use thefuzz to find the best match for the user's query
    best_match = process.extractOne(query, search_keys, scorer=fuzz.WRatio)
    if best_match and best_match[1] >= score_cutoff:
        # If a good match was found (e.g., the key "pm 2.5"), return its corresponding canonical code from the map (e.g., "pm2_5")
        matched_key = best_match[0]
        return pollutant_map[matched_key]
    else:
        return None

def get_pollutant_for_city(city_name, pollutant_name, locations):
    """
    Orchestrates the process of getting a real-time value for a specific pollutant.
    """
    # 1. Use fuzzy matching for the pollutant name
    canonical_pollutant = find_best_pollutant_match(pollutant_name, POLLUTANT_MAP)
    if not canonical_pollutant:
        return f"I'm sorry, I don't have data for a pollutant called '{pollutant_name}'."
    # 2. Find the location's coordinates
    city_name_lower = city_name.lower()
    location_info = next((loc for loc in locations if loc["full_name"].lower() == city_name_lower), None)
    if not location_info:
        return f"Sorry, '{city_name.capitalize()}' is not one of my supported locations."
    # 3. Get real-time pollutant data from the API
    all_pollutants = get_realtime_pollutants(location_info["lat"], location_info["lon"])
    if all_pollutants is None:
        return f"Sorry, I couldn't fetch the live data for {city_name.capitalize()} at the moment."
    # 4. Extract the specific pollutant value
    pollutant_value = all_pollutants.get(canonical_pollutant)
    if pollutant_value is not None:
        # 5. Get the descriptive level
        pollutant_level = get_pollutant_level(canonical_pollutant, pollutant_value)
        unit = "μg/m³"
        # 6. Format the final, more helpful answer
        return (
            f"The current **{pollutant_name.upper()}** level in **{city_name.capitalize()}** "
            f"is **{pollutant_value} {unit}**, which is considered **'{pollutant_level}'**."
        )
    else:
        return f"Sorry, I couldn't find the value for {canonical_pollutant.upper()} in the data for {city_name.capitalize()}."

def get_pollutant_level(pollutant_code, value):
    """
    Classifies a pollutant's numerical value into a descriptive level.
    """
    if pollutant_code not in POLLUTANT_THRESHOLDS:
        return "" # No thresholds defined for this pollutant
    
    thresholds = POLLUTANT_THRESHOLDS[pollutant_code]
    value = float(value)
    
    if value < thresholds["good"]:
        return "Good"
    elif value < thresholds["fair"]:
        return "Fair"
    elif value < thresholds["moderate"]:
        return "Moderate"
    elif value < thresholds["poor"]:
        return "Poor"
    else:
        return "Very Poor"

def get_coordinates(city_name):
    """
    Finds the latitude and longitude for a given city.
    """
    API_KEY = "6a10ef7f599b4187e0cd396738b9fa41" # Replace with your key
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"

    try:
        geo_response = requests.get(geo_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data:
            return f"Sorry, I couldn't find the location: {city_name.capitalize()}."

        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        return f"The coordinates for {city_name.capitalize()} are: Latitude {lat:.4f}, Longitude {lon:.4f}."

    except requests.exceptions.RequestException as e:
        return "Sorry, I'm having trouble connecting to the location service right now."

def find_entity(entities, label):
    """Finds the first entity with a given label."""
    for ent in entities:
        if ent[1] == label:
            return ent
    return None

def contains_fuzzy_keyword(text, keyword, score_cutoff=85):
    """
    Checks if any word in the text is a fuzzy match for the keyword.
    """
    # Split the user's text into individual words
    words = text.lower().split()

    # Check each word for a close match
    for word in words:
        if fuzz.ratio(word, keyword) >= score_cutoff:
            return True
    return False

def find_best_location_match(query, locations_list, score_cutoff=80):
    """
    Finds the best location match using a 3-tiered approach:
    1. Exact match, 2. Known alias match, 3. Fuzzy match.
    """
    query_lower = query.lower()
    location_names = [loc['full_name'] for loc in locations_list]

    # Tier 1: Check for an exact (case-insensitive) match on official names
    for name in location_names:
        if query_lower == name.lower():
            return name

    # Tier 2: Check for a known alias in our dictionary (100% reliable)
    if query_lower in LOCATION_ALIASES:
        return LOCATION_ALIASES[query_lower]

    # Tier 3: Use fuzzy matching as a fallback for other typos
    best_match = process.extractOne(query, location_names, scorer=fuzz.WRatio)
    
    if best_match and best_match[1] >= score_cutoff:
        return best_match[0]
        
    # If all checks fail, return None
    return None

RF_FEATURES = [
    "latitude", "longitude", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "dayofweek", "hour_sin", "hour_cos"
]

def rf_predict_now(features: dict) -> int:
    """Uses the Random Forest model to predict the current AQI."""
    feats = add_time_features(features)
    X = pd.DataFrame([feats]).reindex(columns=RF_FEATURES, fill_value=0)
    return int(aqi_model.predict(X)[0])

def get_aqi_for_city(city_name, locations, aqi_model):
    """Orchestrates the process of getting a real-time AQI prediction for a city."""
    city_name_lower = city_name.lower()
    # Find the location details from your locations data
    location_info = next((loc for loc in locations if loc["full_name"].lower() == city_name_lower), None)
    if not location_info:
        return f"Sorry, '{city_name.capitalize()}' is not one of my supported locations."
    # Get real-time pollutant data
    pollutants = get_realtime_pollutants(location_info["lat"], location_info["lon"])
    if pollutants is None:
        return f"Sorry, I couldn't fetch the live air quality data for {city_name.capitalize()} at the moment."

    # Prepare features and predict
    features_for_prediction = { "latitude": location_info["lat"], "longitude": location_info["lon"], **pollutants }
    predicted_aqi_index = rf_predict_now(features_for_prediction)

    # Also get the 0-500 scale for the livability score calculation
    # We can create a simple mapping for this
    index_to_500_scale = {1: 25, 2: 75, 3: 150, 4: 250, 5: 350}

    # Return a dictionary of results
    return {
        "city": city_name,
        "aqi_1_to_5_scale": predicted_aqi_index,
        "aqi_500_scale": index_to_500_scale.get(predicted_aqi_index, 0),
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

REPRESENTATIVE_WQI = {
    "Vashi": 65,
    "New Panvel": 78,
    "Badlapur": 88,
    "Thane": 75,
    "Airoli": 71,
    "Ambernath": 90,
    "Nerul": 52,
    "Mumbai": 57,
    "Juhu": 54,
    "Worli": 62
}

def get_recommendation_for_aqi_value(value):
    """Returns a specific recommendation based on a numerical AQI value."""
    value = int(value)
    if value <= 50:
        return "With an AQI of {value}, the air is 'Good'. It's a great day for outdoor activities!"
    elif value <= 100:
        return f"With an AQI of {value}, the air is 'Satisfactory'. It's generally safe to be outside."
    elif value <= 200:
        return f"With an AQI of {value}, the air is 'Moderate'. Sensitive groups, like children and the elderly, should reduce prolonged outdoor exertion."
    else: # value > 200
        return f"An AQI of {value} is 'Poor' or worse. It's strongly recommended to avoid outdoor activities, especially for sensitive groups."

# You will also need a helper to format the final AQI string
def get_aqi_response_string(city_name):
    # This is the logic you had before that calls get_aqi_for_city
    # and formats the dictionary into a string.
    aqi_data = get_aqi_for_city(city_name, LOCATIONS_CHATBOT, aqi_model)
    if isinstance(aqi_data, dict):
        return (f"The current predicted AQI for **{aqi_data['city']}** is **{aqi_data['aqi_1_to_5_scale']}** "
                f"on the 1-5 scale, which is considered **'{aqi_data['category']}'**.")
    else:
        return aqi_data

def get_chatbot_response(user_input, predicted_intent, extracted_entities, score, session_data={}):
    """
    Processes user input and session data.
    Returns a dictionary with the response and updated session data.
    """
    # Get current state and context from the session data
    state = session_data.get("state")
    last_topic = session_data.get("last_topic")

    # --- STATE-BASED LOGIC ---
    if state == "waiting_for_city_for_aqi":
        city_name = user_input.strip()
        matched_city = find_best_location_match(city_name, LOCATIONS_CHATBOT)
        if matched_city:
            response_text = get_aqi_response_string(matched_city) 
            return {
                "response": response_text, 
                "session_data": {"state": None, "last_topic": "air"}
            }
        else:
            return {
                "response": f"Sorry, still no match for '{city_name}'. Please ask a new question.",
                "session_data": {"state": None, "last_topic": last_topic}
            }
    
    if state == "waiting_for_city_correction":
        user_reply = user_input.strip().lower()
        if user_reply in ['no', 'nope', 'nevermind', 'cancel']:
            return {
                "response": "Okay, what would you like to ask about instead?",
                "session_data": {"state": None, "last_topic": last_topic}
            }
        
        matched_city = find_best_location_match(user_reply, LOCATIONS_CHATBOT)
        if matched_city:
            response_text = get_aqi_response_string(matched_city)
            return {
                "response": response_text,
                "session_data": {"state": None, "last_topic": "air"}
            }
        else:
            return {
                "response": f"Sorry, I still couldn't find a match for '{user_input.strip()}'.",
                "session_data": {"state": None, "last_topic": last_topic}
            }

    if state == "waiting_for_pollutant":
        # The user's entire message is the pollutant name
        pollutant_name = user_input.strip()
        # Get the city we remembered from the last turn
        city_name = session_data.get('city_to_ask_pollutant_for', 'Ambernath') # Default if needed
        
        response_text = get_pollutant_for_city(city_name, pollutant_name, LOCATIONS_CHATBOT)
        return {"response": response_text, "session_data": {"state": None, "last_topic": "air"}}

    if state == "waiting_for_city_for_pollutant":
        # The user's entire message is the city name
        city_name = user_input.strip()
        # Get the pollutant we remembered from the last turn
        pollutant_name = session_data.get('pollutant_to_ask_city_for')
        
        response_text = get_pollutant_for_city(city_name, pollutant_name, LOCATIONS_CHATBOT)
        return {"response": response_text, "session_data": {"state": None, "last_topic": "air"}}

    # --- IF NO STATE IS SET, PROCEED WITH THE NLP MODEL AS USUAL ---

    # Initialize default values
    response_text = "I'm not sure how to respond to that. Could you try rephrasing?"
    session_data['state'] = None # By default, we clear the state
    
    if predicted_intent in ['get_aqi', 'get_pollutant', 'explain_aqi', 'rank_locations']:
        session_data['last_topic'] = 'air'
    elif predicted_intent in ['get_wqi_start', 'explain_wqi']:
        session_data['last_topic'] = 'water'

    # Main logic block
    if score > 0.65:
        if predicted_intent == 'greet':
            response_text = "Hi! I'm your environmental assistant. How can I help you with air or water quality today?"

        elif predicted_intent == 'goodbye':
            response_text = "You're welcome! Stay safe."

        elif predicted_intent == 'out_of_scope':
            response_text = "I'm sorry, I can only provide information related to air and water quality."

        elif predicted_intent == 'list_locations':
            location_names = sorted([loc.capitalize() for loc in SUPPORTED_LOCATIONS])
            locations_str = ", ".join(location_names)
            response_text = f"I can provide data for the following locations: {locations_str}."

        elif predicted_intent == "get_aqi":
            session_data['last_topic'] = 'air'
            city_entity = find_entity(extracted_entities, 'city')
            
            if not city_entity:
                if any(word in user_input.lower() for word in ["here", "near me"]):
                    # User asked for current location, use the default
                    response_text = get_aqi_response_string("Ambernath")
                    session_data['state'] = None # Clear the state
                else:
                    # The user asked for an AQI but didn't specify where
                    response_text = "Which city's AQI would you like to know?"
                    session_data['state'] = "waiting_for_city_for_aqi" # Set the new state

            user_city_query = city_entity[0]
            matched_city = find_best_location_match(user_city_query, LOCATIONS_CHATBOT)

            if matched_city:
                # If a good match was found, proceed with the prediction
                response_text = get_aqi_response_string(matched_city)
                session_data['state'] = None # Clear the state
            else:
                # If no match was found, ask the user to re-spell it
                    response_text = f"I'm sorry, I couldn't find a supported location matching '{user_city_query.capitalize()}'. Could you please try spelling it again?"
                    session_data['state'] = "waiting_for_city_correction" # Set the correction state

            # Return the final dictionary with the response and updated session data
            return {"response": response_text, "session_data": session_data}

        elif predicted_intent == "get_pollutant":
            session_data['last_topic'] = 'air'
            city_entity = find_entity(extracted_entities, 'city')
            pollutant_entity = find_entity(extracted_entities, 'pollutant')
            
            # Case 1: Pollutant is missing
            if not pollutant_entity:
                response_text = "Which pollutant would you like to know about? (e.g., PM2.5, Ozone, etc.)"
                session_data['state'] = "waiting_for_pollutant"
                # If they mentioned a city, remember it for the next turn
                if city_entity:
                    session_data['city_to_ask_pollutant_for'] = city_entity[0]
                
                return {"response": response_text, "session_data": session_data}

            pollutant_name = pollutant_entity[0]
            
            # Case 2: City is missing
            city_name = None
            if city_entity:
                city_name = city_entity[0]
            elif any(word in user_input.lower() for word in ["here", "near me", "outside"]):
                city_name = "Ambernath"
            
            if not city_name:
                response_text = f"For which city would you like to know the {pollutant_name} level?"
                session_data['state'] = "waiting_for_city_for_pollutant"
                # Remember the pollutant for the next turn
                session_data['pollutant_to_ask_city_for'] = pollutant_name
                
                return {"response": response_text, "session_data": session_data}

            # Case 3: Both city and pollutant are present
            matched_city = find_best_location_match(city_name, LOCATIONS_CHATBOT)
            if not matched_city:
                response_text = f"I'm sorry, I couldn't find a supported location matching '{city_name}'."
            else:
                response_text = get_pollutant_for_city(matched_city, pollutant_name, LOCATIONS_CHATBOT)
            
            session_data['state'] = None # Clear state, as the conversation is complete
            return {"response": response_text, "session_data": session_data}

        elif predicted_intent == 'get_wqi_start':
            # TODO: Add your logic to begin the multi-step WQI conversation
            session_data['last_topic'] = 'water'
            response_text = "The interactive WQI prediction feature is currently under development. Please check back soon!"
            session_data['state'] = None
            return {"response": response_text, "session_data": session_data}

        elif predicted_intent == "explain_aqi":
            session_data['last_topic'] = 'air'
            session_data['state'] = None
            
            aqi_value_entity = find_entity(extracted_entities, 'aqi_value')
            level_entity = find_entity(extracted_entities, 'level')

            if aqi_value_entity:
                try:
                    value = int(aqi_value_entity[0])
                    response_text = ""

                    if 1 <= value <= 5:
                        # If the number is 1-5, explain the dashboard scale
                        category = map_dashboard_index_to_category(value).lower()
                        response_text = (
                            f"An index of '{value}' on the dashboard's simple 1-to-5 scale corresponds to **'{category}'**.\n\n"
                            "This index is provided by our data source (OpenWeather API)."
                        )
                    else:
                        # Otherwise, explain the official 0-500 scale
                        category = map_aqi_to_category(value).lower()
                        explanation = AIR_KB.get(category, {}).get("definition", "This value falls within the official AQI scale.")
                        response_text = f"An AQI of **{value}** is considered **'{category.capitalize()}'**. {explanation}"
                    return {"response": response_text, "session_data": session_data}
                except ValueError:
                    return {"response": "I'm sorry, I didn't understand that number. Could you please try again?", "session_data": session_data}
            
            # Handle level names
            elif level_entity:
                level = level_entity[0].lower()
                response_text = AIR_KB.get(level, {}).get("definition", f"I don't have a specific definition for a '{level}' AQI, but it generally indicates a need for caution.")
                return {"response": response_text, "session_data": session_data}
            # Handle general keywords
            else:
                user_text = user_input.lower()
                response_text = ""
                if "color" in user_text:
                    response_text = AIR_KB.get("colors_explanation")
                elif "level" in user_text or "categor" in user_text:
                    response_text = AIR_KB.get("levels_explanation")
                else:
                    response_text = AIR_KB.get("default_explanation")
            
            return {"response": response_text, "session_data": session_data}
        
        elif predicted_intent == 'explain_wqi':
            session_data['last_topic'] = 'water'
            session_data['state'] = None
            
            wqi_value_entity = find_entity(extracted_entities, 'wqi_value')
            level_entity = find_entity(extracted_entities, 'level')
            response_text = ""

            # Priority 1: Handle specific numerical values like "50"
            if wqi_value_entity:
                try:
                    value = int(wqi_value_entity[0])
                    category = map_wqi_to_category(value).lower()
                    
                    # Look up the definition in the water knowledge base
                    explanation = WATER_KB.get(category, {}).get("definition", "This value falls within the official WQI scale.")
                    response_text = f"A WQI of **{value}** is considered **'{category.capitalize()}'**. {explanation}"
                    
                except ValueError:
                    response_text = "I'm sorry, I didn't understand that number. Could you please try again?"
            
            # Handle level names
            elif level_entity:
                level = level_entity[0].lower()
                response_text = WATER_KB.get(level, {}).get("definition", f"I don't have a specific definition for a '{level}' WQI.")
            
            # Priority 3: Default WQI explanation if no specific entities are found
            else:
                response_text = WATER_KB.get("default_explanation", "The Water Quality Index (WQI) is a score from 0-100 that indicates the health of a water body.")

            return {"response": response_text, "session_data": session_data}

        elif predicted_intent == "ask_general_knowledge":
            user_text = user_input.lower()
            city_entity = find_entity(extracted_entities, 'city')
            term_entity = find_entity(extracted_entities, 'term')
            topic_entity = find_entity(extracted_entities, 'topic')
            level_entity = find_entity(extracted_entities, 'level')
            
            response_text = "I can answer factual questions about AQI and WQI. What would you like to know?" # Default
            
            # --- 1. Determine the Primary Context (Air vs. Water) ---
            topic_context = None
            if any(word in user_text for word in ['aqi', 'air', 'smog', 'pollutant', 'trees']):
                topic_context = 'air'
                session_data['last_topic'] = 'air'
            elif any(word in user_text for word in ['wqi', 'water', 'bod', 'ph', 'tds', 'drinking']):
                topic_context = 'water'
                session_data['last_topic'] = 'water'
            else:
                # Fallback to conversation history if no explicit keywords are found
                topic_context = session_data.get('last_topic', 'air')
            
            # --- 2. Select the Correct Knowledge Base ---
            knowledge_base_to_use = AIR_KB if topic_context == 'air' else WATER_KB
            
            # --- 3. Prioritized Logic Chain ---
            # Priority A: Handle specific, high-priority keyword queries
            if "coordinate" in user_text or "latitude" in user_text:
                if city_entity:
                    response_text = get_coordinates(city_entity[0])
                else:
                    response_text = "Which city's coordinates would you like to know?"

            elif (contains_fuzzy_keyword(user_text, "how") and (("aqi" in user_text) or ("air" in user_text)) and (contains_fuzzy_keyword(user_text, "calculated") or contains_fuzzy_keyword(user_text, "measured"))):
                response_text = AIR_KB.get("how_aqi_calculated")
            elif (contains_fuzzy_keyword(user_text, "how") and (("wqi" in user_text) or ("water" in user_text)) and (contains_fuzzy_keyword(user_text, "calculated") or contains_fuzzy_keyword(user_text, "measured"))):
                response_text = WATER_KB.get("how_wqi_calculated")
            elif (contains_fuzzy_keyword(user_text, "source") or contains_fuzzy_keyword(user_text, "cause")) and "air pollution" in user_text:
                response_text = AIR_KB.get("sources_air")
            elif (contains_fuzzy_keyword(user_text, "source") or contains_fuzzy_keyword(user_text, "cause")) and "water pollution" in user_text:
                response_text = WATER_KB.get("sources_water")
            elif (contains_fuzzy_keyword(user_text, "standards") and  "aqi" in user_text):
                response_text = AIR_KB.get("aqi_standards_india")
            elif 'trees' in user_text:
                response_text = AIR_KB.get("trees")
            elif 'cities' in user_text and 'water' in user_text:
                response_text = WATER_KB.get("cities_water")
            elif 'bottled water' in user_text:
                response_text = WATER_KB.get("bottled_water")
            elif 'safe' in user_text and 'drinking' in user_text:
                response_text = WATER_KB.get("safe_drinking_level")
            elif 'improve' in user_text and 'at home' in user_text:
                response_text = WATER_KB.get("improve_at_home")

            # --- Main entity-based logic ---
            else:
                main_topic = None
                if term_entity: main_topic = term_entity[0].lower()
                elif topic_entity: main_topic = topic_entity[0].lower()
                if main_topic:
                    knowledge_base_to_use = AIR_KB
                    if any(word in main_topic for word in ['wqi', 'water', 'bod', 'ph', 'tds']) or session_data.get('last_topic') == 'water':
                        knowledge_base_to_use = WATER_KB
                        
                    if main_topic in knowledge_base_to_use:
                        knowledge_entry = knowledge_base_to_use[main_topic]
                        # Check if the knowledge entry is a detailed topic (a dictionary)
                        if isinstance(knowledge_entry, dict):
                            if "effect" in user_text or "health" in user_text:
                                response_text = knowledge_entry.get("effects", "I don't have specific information on the health effects for that topic.")
                            else:
                                response_text = knowledge_entry.get("info",  f"I have some information about {main_topic}, but not the specific details you're looking for.")
                        else:
                            response_text = knowledge_entry

            session_data['state'] = None
            return {"response": response_text, "session_data": session_data}
        
        elif predicted_intent == 'give_recommendations':
            # 1. Extract all possible entities
            aqi_value_entity = find_entity(extracted_entities, 'aqi_value')
            level_entity = find_entity(extracted_entities, 'level')
            topic_entity = find_entity(extracted_entities, 'topic')
            term_entity = find_entity(extracted_entities, 'term')
            user_text = user_input.lower()

            # 2. Determine the Context
            topic_context = 'air' # Default
            # Check for explicit keywords
            if any(word in user_text for word in ['water', 'wqi', 'ph', 'tds', 'bod', 'cod', 'drinking']):
                topic_context = 'water'
            # If no keywords, check the conversation history
            elif session_data.get('last_topic') == 'water':
                topic_context = 'water'

            # 3. Select the correct Knowledge Base and default response
            if topic_context == 'air':
                knowledge_base_to_use = AIR_KB
                default_response = AIR_KB.get('default_recommendation', "General air quality advice...")
            else: # topic_context == 'water'
                knowledge_base_to_use = WATER_KB
                default_response = WATER_KB.get('default_recommendation', "General water quality advice...")

            response_text = default_response
            
            # 4. Find the Answer in the Correct Knowledge Base
            # Priority 1: Handle specific numerical AQI values (always air-related)
            if aqi_value_entity:
                try:
                    value = int(aqi_value_entity[0])
                    response_text = get_recommendation_for_aqi_value(value)
                except ValueError:
                    response_text = default_response

            else:
                # Priority 2: Handle descriptive levels, topics, or terms
                key_entity = level_entity or topic_entity or term_entity
                if key_entity:
                    key = key_entity[0].lower().replace(" ", "_")
                    if key in knowledge_base_to_use and isinstance(knowledge_base_to_use[key], dict):
                        response_text = knowledge_base_to_use[key].get("recommendation", default_response)
                    # Check for direct key match
                    elif key in knowledge_base_to_use:
                        response_text = knowledge_base_to_use[key]

            # Priority 3: Fallback to the default response for the determined context
            session_data['state'] = None # This intent doesn't require a follow-up state
            return {"response": response_text, "session_data": session_data}
        
        elif predicted_intent == 'rank_locations':
            session_data['state'] = None
            # 1. Extract all city entities
            cities_to_compare = [ent[0] for ent in extracted_entities if ent[1] == 'city']

            if len(cities_to_compare) == 1:
                # If only one city was found, treat it as a 'get_aqi' request
                city_name = find_best_location_match(cities_to_compare[0], LOCATIONS_CHATBOT)
                if city_name:
                    response_text = get_aqi_response_string(city_name)
                    session_data['last_topic'] = 'air'
                    return {"response": response_text, "session_data": session_data}
                else:
                    response_text = f"I'm sorry, I couldn't find the location '{cities_to_compare[0]}'."
                    return {"response": response_text, "session_data": session_data}

            elif len(cities_to_compare) < 2:
                response_text = "Please provide at least two locations for me to compare for livability."
                return {"response": response_text, "session_data": session_data}

            # 2. Get AQI and WQI data for each city
            results = []
            for city_query in cities_to_compare:
                matched_city = find_best_location_match(city_query, LOCATIONS_CHATBOT)
                if matched_city:
                    # Get the structured AQI data
                    aqi_data = get_aqi_for_city(matched_city, LOCATIONS_CHATBOT, aqi_model)
                    if isinstance(aqi_data, dict):
                        # Get the representative WQI score
                        wqi_score = REPRESENTATIVE_WQI.get(matched_city, 0) # Default to a high score if not found
                        
                        # Convert 0-500 AQI (lower is better) to a 0-100 score (higher is better)
                        aqi_500_value = aqi_data['aqi_500_scale']
                        air_score = max(0, 100 - (aqi_500_value / 500) * 100)
                        
                        # WQI is already a 0-100 score (higher is better)
                        water_score = wqi_score
                        
                        # Average the two scores for a final livability score
                        livability_score = (air_score + water_score) / 2
                        
                        results.append({
                            'city': matched_city,
                            'aqi': aqi_data['aqi_1_to_5_scale'],
                            'wqi': wqi_score,
                            'score': livability_score
                        })
            
            if len(results) < 2:
                response_text = "I couldn't find reliable data for at least two of the locations you mentioned."
                return {"response": response_text, "session_data": session_data}
            
            # 3. Sort the results by the livability score (higher is better)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # 4. Build the final, helpful response
            best_location = results[0]
            response_text = (
                f"Based on a combination of current air and water quality, "
                f"**{best_location['city']}** appears to be the better option.<br><br>"
                "Here is the full comparison (higher scores are better):<br>"
            )

            for res in results:
                response_text += f"- **{res['city']}**: AQI Index: {res['aqi']}, Representative WQI: {res['wqi']:.0f}<br>"
            return {"response": response_text, "session_data": session_data}

    return {"response": response_text, "session_data": session_data}
# ---------------------------------------- END OF CHATBOT ---------------------------------------

locations_df = pd.read_csv(LOCATIONS_CSV)
LOCATIONS = [
    {"id": i, "full_name": row["location_name"], "lat": float(row["latitude"]), "lon": float(row["longitude"])}
    for i, row in locations_df.iterrows()
]

# =========================
# 2) Helpers
# =========================

def aqi_color(aqi_class: int) -> str:
    """Returns a color hex code based on AQI class."""
    return {
        1: "#00B050", 2: "#92D050", 3: "#FFFF00",
        4: "#FF9900", 5: "#FF0000"
    }.get(int(aqi_class), "#7f8c8d")

def wqi_color(wqi_value: float) -> str:
    """Returns a color hex code based on WQI score."""
    wqi_value = int(wqi_value)
    if wqi_value >= 63: return "#2ecc71" # Green (Good)
    elif wqi_value >= 50: return "#f1c40f" # Yellow (Medium)
    elif wqi_value >= 38: return "#e67e22" # Orange (Bad)
    else: return "#e74c3c" # Red (Very Bad)

def get_all_water_data():
    """Predicts the WQI for all water locations using the trained ML model and caches the results."""
    global CACHE_WATER

    # Check if we have valid, recent data in our cache
    if CACHE_WATER["data"] and CACHE_WATER["timestamp"]:
        age = datetime.now() - CACHE_WATER["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            print("Serving water data from cache...")
            return CACHE_WATER["data"]
    
    print("Water cache is old or empty. Processing data from CSV...")
    results = []
    try:
        water_df = pd.read_csv("water_rep_data.csv")
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
            
            # 2. Generate SHAP and LIME explanations
            shap_explanation = get_wqi_shap_explanation(input_scaled)
            lime_explanation = get_wqi_lime_explanation(input_scaled)
            
            # 4. Get classification and recommendations based on the PREDICTED score
            classification = get_wqi_category_and_remark(predicted_wqi)
            recommendations = get_water_recommendations(predicted_wqi, shap_explanation)
            
            # 5. Bundle the results
            results.append({
                "id": int(i),
                "full_name": str(row["location_name"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "wqi": float(predicted_wqi),
                "classification": classification["classification"],
                "remark": classification["remark"],
                "recommendation": recommendations,
                "lime_explanation": lime_explanation,
                "shap_explanation": shap_explanation,
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

def get_all_water_forecasts():
    """Generates future WQI predictions for all water locations."""
    global CACHE_WATER_FORECAST
    # Check if we have valid, recent data in our cache
    if CACHE_WATER_FORECAST["data"] and CACHE_WATER_FORECAST["timestamp"]:
        age = datetime.now() - CACHE_WATER_FORECAST["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            print("Serving data from cache...")
            return CACHE_WATER_FORECAST["data"]

    print("Water forecast cache is old or empty. Predicting new data...")
    results = []
    try:
        water_base_df = pd.read_csv("water_rep_data.csv") # Your CSV with base parameters
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

def build_full_water_dashboard_map(water_data):
    """Builds the full-screen map for the water dashboard page, including a forecast heatmap."""
    if not water_data: 
        return ""
    
    center_lat = float(np.mean([p["lat"] for p in water_data]))
    center_lon = float(np.mean([p["lon"] for p in water_data]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

    # --- 1. Add Circle Markers for Current WQI (Same as before) ---
    for p in water_data:
        col = wqi_color(p["wqi"])
        tooltip_text = f"<b>{p['full_name']}</b><br>WQI (Predicted): <b>{p['wqi']:.0f}</b> ({p['classification']})"
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=9, color=col, fill=True,
            fill_color=col, fill_opacity=0.9, tooltip=tooltip_text, popup=tooltip_text
        ).add_to(m)
        
    # --- 2. NEW: Add the Animated Forecast Heatmap ---
    time_slices = []
    time_labels = []
    now = datetime.now()

    for h in WQI_HORIZONS: # Use your water horizons (e.g., [1, 2, 3, 4, 5, 6])
        slice_points = []
        for p in water_data:
            forecast = p.get("forecast", {}) 
            # Get the forecast value using the correct key format (e.g., '1_month')
            wqi_val = forecast.get(f"{h}_month")
            if wqi_val: 
                slice_points.append([p["lat"], p["lon"], wqi_val])
        
        if slice_points:
            time_slices.append(slice_points)
            # Create a label for the future month
            future_date = now + relativedelta(months=int(h))
            time_labels.append(future_date.strftime(f"t+{h}m (%b %Y)"))

    if time_slices:
        HeatMapWithTime(
            data=time_slices, 
            index=time_labels, 
            auto_play=False, 
            max_opacity=0.8,
            radius=27, 
            use_local_extrema=True, 
            name="Forecast Heatmap (Next 6 Months)"
        ).add_to(m)
        
    LayerControl().add_to(m)
    return m._repr_html_()

def get_wqi_lime_explanation(scaled_features_df: pd.DataFrame) -> list:
    """Generates LIME explanation for the WQI regressor's prediction."""
    instance_to_explain = scaled_features_df[0] # LIME needs a 1D array
    
    explanation = wqi_lime_explainer.explain_instance(
        instance_to_explain,
        wqi_model.predict, # For regressors, just use .predict
        num_features=5
    )
    return explanation.as_list()

def get_all_water_data():
    """Predicts the WQI for all water locations using the trained ML model and caches the results."""
    global CACHE_WATER

    # Check if we have valid, recent data in our cache
    if CACHE_WATER["data"] and CACHE_WATER["timestamp"]:
        age = datetime.now() - CACHE_WATER["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            print("Serving water data from cache...")
            return CACHE_WATER["data"]
    
    print("Water cache is old or empty. Processing data from CSV...")
    results = []
    try:
        water_df = pd.read_csv("water_rep_data.csv")
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
            
            # 2. Generate SHAP and LIME explanations
            shap_explanation = get_wqi_shap_explanation(input_scaled)
            lime_explanation = get_wqi_lime_explanation(input_scaled)
            
            # 4. Get classification and recommendations based on the PREDICTED score
            classification = get_wqi_category_and_remark(predicted_wqi)
            recommendations = get_water_recommendations(predicted_wqi, shap_explanation)
            
            # 5. Bundle the results
            results.append({
                "id": int(i),
                "full_name": str(row["location_name"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "wqi": float(predicted_wqi),
                "classification": classification["classification"],
                "remark": classification["remark"],
                "recommendation": recommendations,
                "lime_explanation": lime_explanation,
                "shap_explanation": shap_explanation,
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

# 1. Load data once when the app starts, inside the app context
with app.app_context():
    load_all_models_and_data()
    
    ALL_WATER_DATA = get_all_water_data()
    ALL_WATER_FORECASTS = get_all_water_forecasts()

    # Create dictionaries for fast lookups using the location ID
    WATER_DATA_DICT = {str(loc.get("id")): loc for loc in ALL_WATER_DATA}
    WATER_FORECAST_DICT = {str(loc.get("id")): loc for loc in ALL_WATER_FORECASTS}

def build_generic_map(map_data, value_key, color_function):
    """
    Builds a generic Folium map for either air or water data.
    """
    if not map_data:
        return ""
    
    center_lat = float(np.mean([p["lat"] for p in map_data]))
    center_lon = float(np.mean([p["lon"] for p in map_data]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

    for p in map_data:
        value = p[value_key]
        CircleMarker(
            location=[p["lat"], p["lon"]], 
            radius=16, 
            color=color_function(value),
            fill=True, 
            fill_color=color_function(value), 
            fill_opacity=0.6, 
            tooltip=p["full_name"]
        ).add_to(m)
    
    return m._repr_html_()

def add_time_features(d: dict) -> dict:
    """Adds time-based features to a dictionary."""
    out = d.copy()
    now = datetime.now()
    hour = out.get("hour", now.hour)
    out["hour"] = hour
    out["day"] = now.day
    out["month"] = now.month
    out["dayofweek"] = now.weekday()
    out["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    out["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    return out

def get_realtime_pollutants(lat: float, lon: float) -> dict:
    """Fetches real, live air pollution data from the OpenWeather API."""
    """Generates varied data for predictions."""
    api_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {'lat': lat, 'lon': lon, 'appid': API_KEY}

    default_pollutants = {"co": 0, "no2": 0, "o3": 0, "so2": 0, "pm2_5": 0, "pm10": 0}

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data and 'list' in data and data['list']:
            return data['list'][0]['components']
        else:
            return default_pollutants

    except requests.exceptions.RequestException as e:
        print(f"API Error fetching OpenWeather data for lat={lat}, lon={lon}: {e}")
        return default_pollutants

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

def predict_all_locations():
    """Predicts current and future AQI for all hardcoded locations."""
    global CACHE

    # Check if we have valid, recent data in our cache
    if CACHE["data"] and CACHE["timestamp"]:
        age = datetime.now() - CACHE["timestamp"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            print("Serving data from cache...")
            return CACHE["data"]

    # If cache is empty or old, fetch new data from the API
    print("Cache is old or empty. Fetching new data from API...")
    results = []
    for loc in LOCATIONS:
        try:
            base_pollutants = get_realtime_pollutants(loc["lat"], loc["lon"])
            feat_dict = {"latitude": loc["lat"], "longitude": loc["lon"], **base_pollutants}
            timed_feat = add_time_features(feat_dict)
            X_raw = pd.DataFrame([timed_feat]).reindex(columns=MODEL_FEATURES, fill_value=0)
            
            aqi_now = int(final_model.predict(X_raw)[0])
            forecast = lgb_predict_future(timed_feat)
            shap_explanation = get_shap_explanation(X_raw)
            lime_explanation = get_lime_explanation(X_raw)
            recommendations = get_recommendations(aqi_now, shap_explanation)
            
            results.append({
                "id": loc["id"], "full_name": loc["full_name"], "lat": loc["lat"], "lon": loc["lon"],
                "aqi_now": aqi_now, "forecast": forecast, "pollutants": base_pollutants,
                "shap_explanation": shap_explanation,
                "lime_explanation": lime_explanation,
                "recommendations": recommendations
            })
            time.sleep(0.1)
        except Exception as e:
            print(f"!!! FAILED TO PROCESS LOCATION: {loc.get('full_name', 'N/A')} !!!")
            print(f"ERROR: {e}")

    # Save the new results and the current time to the cache
    CACHE["data"] = results
    CACHE["timestamp"] = datetime.now()
    return results
    
def lgb_predict_future(features: dict) -> dict:
    feats = add_time_features(features)
    X = pd.DataFrame([feats]).reindex(columns=lgb_features, fill_value=0)
    preds = lgb_model.predict(X)[0]
    return {f"{h}h": int(p) for h, p in zip(lgb_horizons, preds)}

# def build_map_for_card(all_preds: list):
#     """Builds a static Folium map for the landing page card."""
#     # all_preds = predict_all_locations()
#     if not all_preds: 
#         return ""
#     center_lat = float(np.mean([p["lat"] for p in all_preds]))
#     center_lon = float(np.mean([p["lon"] for p in all_preds]))
#     m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

#     for p in all_preds:
#         CircleMarker(
#             location=[p["lat"], p["lon"]], radius=16, color=aqi_color(p["aqi_now"]),
#             fill=True, fill_color=aqi_color(p["aqi_now"]), fill_opacity=0.6, tooltip=p["full_name"]
#         ).add_to(m)
#     return m._repr_html_()

def build_full_dashboard_map():
    """Builds the full-screen map with heatmap for the dashboard page."""
    all_preds = predict_all_locations()
    if not all_preds: 
        return ""
    center_lat = float(np.mean([p["lat"] for p in all_preds]))
    center_lon = float(np.mean([p["lon"] for p in all_preds]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

    for p in all_preds:
        col = aqi_color(p["aqi_now"])
        tooltip_text = f"<b>{p['full_name']}</b><br>AQI (now): <b>{p['aqi_now']}</b>"
        popup_html = f"""
        <h4>{p['full_name']}</h4>
        <b>Current AQI: {p['aqi_now']}</b><br>
        72h Forecast Trend: <b>{p['forecast']}</b>
        <hr style='margin: 5px 0;'>
        <i>See the chart below for hourly details.</i>
        """
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=9, color=col, fill=True,
            fill_color=col, fill_opacity=0.9, tooltip=tooltip_text, popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)
    time_slices, time_labels, now = [], [], datetime.now()
    for h in lgb_horizons:
        slice_points = []
        for p in all_preds:
            aqi_val = p["forecast"].get(f"{h}h")
            if aqi_val: slice_points.append([p["lat"], p["lon"], aqi_val])
        if slice_points:
            time_slices.append(slice_points)
            time_labels.append((now + timedelta(hours=int(h))).strftime("t+%Hh (%a)"))
    if time_slices:
        HeatMapWithTime(
            data=time_slices, index=time_labels, auto_play=False, max_opacity=0.8,
            radius=27, use_local_extrema=True, name="Forecast Heatmap (3–72h)"
        ).add_to(m)
    LayerControl().add_to(m)
    return m._repr_html_()

def build_plotly_series(pred_for_loc: dict):
    """Formats a single location's forecast data for Plotly chart."""
    if not pred_for_loc: 
        return []
    series = []
    now = datetime.now()
    for h, aqi in pred_for_loc["forecast"].items():
        ts = (now + timedelta(hours=int(h.replace("h", "")))).isoformat()
        series.append({"ts": ts, "aqi": int(aqi), "h": h})
    return series

# =========================
# 3) Routes
# =========================
@app.route("/")
def landing_page(): 
    return render_template("front_page.html")

@app.route("/dashboard")
def dashboard():
    # --- Get Air Data ---
    all_air_preds = predict_all_locations() or []
    air_card_map = build_generic_map(all_air_preds, 'aqi_now', aqi_color)
    initial_air_data = all_air_preds[0] if all_air_preds else None

    # --- Get Water Data ---
    all_water_data = get_all_water_data() or []
    water_card_map = build_generic_map(all_water_data, 'wqi', wqi_color)
    initial_water_data = all_water_data[0] if all_water_data else None

    return render_template(
        "index.html",
        # Air Data
        air_map_html=air_card_map,
        air_locations_data=all_air_preds,
        initial_air_data=initial_air_data,
        # Water Data
        water_map_html=water_card_map,
        water_locations_data=all_water_data,
        initial_water_data=initial_water_data
    )

@app.route("/api/predict_wqi", methods=["POST"])
def api_predict_wqi():
    data = request.json
    try:
        location_id = int(data.get("location_id"))
        location_info = get_all_water_data()[location_id]
        
        features = {
            "pH": float(data.get("ph")),
            "dissolvedoxygen": float(data.get("dissolvedoxygen")),
            "bod": float(data.get("bod")),
            "cod": float(data.get("cod")),
            "nitrate": float(data.get("nitrate")),
            "FecalColiform": float(data.get("fecalcoliform")),
            "Year": datetime.now().year,
            "Month": int(data.get("month")),
            "latitude": location_info["lat"],
            "longitude": location_info["lon"]
        }
        
        input_df = pd.DataFrame([features]).reindex(columns=WQI_FEATURES, fill_value=0)
        input_scaled = wqi_scaler.transform(input_df)
        
        # --- FIX IS HERE: Convert the NumPy float to a standard Python float ---
        predicted_wqi_score = float(wqi_model.predict(input_scaled)[0])
        
        wqi_classification = get_wqi_category_and_remark(predicted_wqi_score)
        
        return jsonify({
            "ok": True,
            "wqi_score": f"{predicted_wqi_score:.2f}",
            "classification": wqi_classification["classification"],
            "remark": wqi_classification["remark"]
        })

    except Exception as e:
        print(f"Error during WQI prediction: {e}")
        return jsonify({"ok": False, "error": "An error occurred during prediction."}), 500

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """API endpoint for the chatbot to receive and respond to messages."""
    load_all_models_and_data()
    user_input = request.json.get("message", "")
    session_data = request.json.get("session_data", {}) 
    if not user_input:
        return jsonify({"response": "Sorry, I didn't receive a message."})
    
    # 1. Get the intent from the Textcat model
    doc_textcat = textcat_model(user_input)
    predicted_intent = max(doc_textcat.cats, key=doc_textcat.cats.get)
    score = doc_textcat.cats[predicted_intent]

    # 2. Get the entities from the NER model
    doc_ner = ner_model(user_input)
    extracted_entities = [(ent.text, ent.label_) for ent in doc_ner.ents]

    # # --- Debug Prints for Your Terminal ---
    # print("\n--- CHATBOT DEBUG ---")
    # print(f"User Input: '{user_input}'")
    # print(f"--> Predicted Intent: {predicted_intent} (Score: {score:.2f})")
    # if extracted_entities:
    #     print(f"--> Extracted Entities: {extracted_entities}")
    # print("---------------------\n")
    
    # Pass the NLP results to your main logic function
    response_data = get_chatbot_response(
        user_input,
        predicted_intent,
        extracted_entities,
        score,
        session_data
    )

    return jsonify(response_data)

@app.route("/air_dashboard")
def air_dashboard():
    all_preds = predict_all_locations()
    folium_html = build_full_dashboard_map()
    default_loc = all_preds[0] if all_preds else None
    if default_loc is None:
        chart_series, default_location_name = [], "No Locations"
    else:
        chart_series = build_plotly_series(default_loc)
        default_location_name = default_loc["full_name"]
    return render_template(
        "air_dashboard.html", map_html=folium_html, locations=all_preds,
        default_location_name=default_location_name, chart_series=json.dumps(chart_series),
        initial_details=default_loc
    )

@app.route("/api/forecast_series")
def api_forecast_series():
    id = request.args.get("id", "").strip()
    if not id: 
        return jsonify({"ok": False, "error": "Missing 'id'"}), 400
    all_preds = predict_all_locations()
    loc = next((p for p in all_preds if str(p["id"]) == id), None)
    if not loc: 
        return jsonify({"ok": False, "error": "Location not found"}), 404
    return jsonify({"ok": True, "name": loc['full_name'], "series": build_plotly_series(loc)})


@app.route("/water_dashboard")
def water_dashboard(): 
    """Full water quality dashboard page with forecast."""
    # Note: Use the cached data we defined earlier
    all_water_data = ALL_WATER_DATA
    water_forecast_dict = WATER_FORECAST_DICT
    print(f"[DEBUG] Water data count: {len(all_water_data)}, Forecast count: {len(water_forecast_dict)}")

    # Attach forecast to each location using its ID (robust)
    for loc in all_water_data:
        forecast_data = water_forecast_dict.get(str(loc.get("id")))
        loc["forecast"] = forecast_data.get("forecast", {}) if forecast_data else {}

    folium_html = build_full_water_dashboard_map(all_water_data)
    default_loc = all_water_data[0] if all_water_data else None
    
    if default_loc is None:
        chart_series, default_location_name = [], "No Locations"
    else:
        chart_series = build_wqi_plotly_series(default_loc['forecast']) or []
        default_location_name = default_loc["full_name"]
    
    print("Default Location Forecast:", default_loc.get("forecast") if default_loc else None)
    print("Chart Series:", chart_series)

    return render_template(
        "water_dashboard.html", 
        map_html=folium_html, 
        locations=all_water_data,
        default_location_name=default_location_name, 
        chart_series=chart_series,
        initial_details=default_loc
    )

@app.route("/api/wqi_forecast_series")
def api_wqi_forecast_series():
    """API endpoint to fetch WQI forecast series for a specific location."""
    id_str = request.args.get("id", "").strip()
    if not id_str: 
        return jsonify({"ok": False, "error": "Missing 'id'"}), 400
        
    # Find forecast data from the dictionary (very fast)
    loc = WATER_FORECAST_DICT.get(id_str)
    
    if not loc: 
        return jsonify({"ok": False, "error": "Location not found"}), 404
        
    return jsonify({"ok": True, "name": loc['full_name'], "series": build_wqi_plotly_series(loc['forecast'])})

@app.route("/api/wqi_location_details")
def api_wqi_location_details():
    """API endpoint for WQI recommendations and LIME."""
    id_str = request.args.get("id", "").strip()
    if not id_str:
        return jsonify({"ok": False, "error": "Missing 'id'"})
        
    # Find location data from the dictionary (very fast)
    loc = WATER_DATA_DICT.get(id_str)
    
    if not loc:
        return jsonify({"ok": False, "error": "Location not found"})
        
    return jsonify({
        "ok": True,
        "recommendation": loc.get("recommendation"),
        "lime_explanation": loc.get("lime_explanation")
    })

if __name__ == "__main__":
    app.run(debug=True)
