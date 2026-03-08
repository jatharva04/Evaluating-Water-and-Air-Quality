import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Cache Settings & API Keys ---
CACHE = { "data": None, "timestamp": None }
CACHE_WATER = { "data": None, "timestamp": None }
CACHE_WATER_FORECAST = {"data": None, "timestamp": None}
CACHE_DURATION_MINUTES = 30
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- Model Features and Horizons ---
WQI_HORIZONS = [1, 2, 3, 4, 5, 6] # In months
LOCATIONS_CSV = "data/Locations.csv"
MODEL_FEATURES = [
    "latitude", "longitude", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "dayofweek", "hour_sin", "hour_cos"
]
RF_FEATURES = MODEL_FEATURES.copy()

# --- Geographic & Chatbot Knowledge ---
SUPPORTED_LOCATIONS = {"panvel", "vashi", "airoli", "nerul", "thane", "worli", "juhu", "versova", "badlapur", "ambernath", "powai"}

# Load Locations
_chatbot_locations_df = pd.read_csv("data/locations_chatbot.csv")
LOCATIONS_CHATBOT = [
    {"id": i, "full_name": row["location_name"], "lat": float(row["latitude"]), "lon": float(row["longitude"])}
    for i, row in _chatbot_locations_df.iterrows()
]

_locations_df = pd.read_csv(LOCATIONS_CSV)
LOCATIONS = [
    {"id": i, "full_name": row["location_name"], "lat": float(row["latitude"]), "lon": float(row["longitude"])}
    for i, row in _locations_df.iterrows()
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

POLLUTANT_THRESHOLDS = {
    "so2":   {"good": 20, "fair": 80, "moderate": 250, "poor": 350},
    "no2":   {"good": 40, "fair": 70, "moderate": 150, "poor": 200},
    "pm10":  {"good": 20, "fair": 50, "moderate": 100, "poor": 200},
    "pm2_5": {"good": 10, "fair": 25, "moderate": 50, "poor": 75},
    "o3":    {"good": 60, "fair": 100, "moderate": 140, "poor": 180},
    "co":    {"good": 4400, "fair": 9400, "moderate": 12400, "poor": 15400}
}

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
