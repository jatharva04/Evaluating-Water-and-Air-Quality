import os
import math
import json
import requests
import time
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib

import folium
from folium import Map, CircleMarker, LayerControl
from folium.plugins import HeatMapWithTime

app = Flask(__name__)

# This dictionary will store our API results temporarily.
CACHE = { "data": None, "timestamp": None }
CACHE_DURATION_MINUTES = 15

API_KEY = "6a10ef7f599b4187e0cd396738b9fa41"

# =========================
# 1) Load models and data
# =========================
RF_PATH = os.path.join("models", "random_forest_aqi.pkl")
LGB_BUNDLE_PATH = os.path.join("models", "lightgbm_multi.pkl")
LOCATIONS_CSV = "locations.csv"

# Check if files exist before loading
if not os.path.exists(RF_PATH) or not os.path.exists(LGB_BUNDLE_PATH) or not os.path.exists(LOCATIONS_CSV):
    raise FileNotFoundError("One or more required files (models or locations.csv) are missing.")

rf_model = joblib.load(RF_PATH)
lgb_bundle = joblib.load(LGB_BUNDLE_PATH)
lgb_model = lgb_bundle["model"]
lgb_features = lgb_bundle["features"]
lgb_horizons = lgb_bundle["horizons"]

locations_df = pd.read_csv(LOCATIONS_CSV)
LOCATIONS = [
    # The 'id' will be a URL-safe unique identifier.
    # The 'full_name' is the full name from your CSV, unaltered.
    {"id": i, "full_name": row["location_name"], "lat": float(row["latitude"]), "lon": float(row["longitude"])}
    for i, row in locations_df.iterrows()
]

# Required features for the models
RF_FEATURES = [
    "latitude", "longitude", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "dayofweek", "hour_sin", "hour_cos"
]

LGB_FEATURES = lgb_features

# =========================
# 2) Helpers
# =========================
def aqi_color(aqi_class: int) -> str:
    """Returns a color hex code based on AQI class."""
    return {
        1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22",
        4: "#e74c3c", 5: "#8e44ad"
    }.get(int(aqi_class), "#7f8c8d")

def add_time_features(d: dict) -> dict:
    """Adds time-based features (day, month, hour, dayofweek, etc.) to a dictionary."""
    out = d.copy()
    now = datetime.now()
    hour = out.get("hour", now.hour)
    day = out.get("day", now.day)
    month = out.get("month", now.month)
    
    out["hour"] = hour
    out["day"] = day
    out["month"] = month
    out["dayofweek"] = now.weekday()
    out["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    out["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    return out

def get_realtime_pollutants(lat: float, lon: float) -> dict:
    """Fetches real, live air pollution data from the OpenWeather API."""
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


def rf_predict_now(features: dict) -> int:
    """Uses the Random Forest model to predict the current AQI."""
    feats = add_time_features(features)
    X = pd.DataFrame([feats]).reindex(columns=RF_FEATURES, fill_value=0)
    return int(rf_model.predict(X)[0])

def lgb_predict_future(features: dict) -> dict:
    """Uses the LightGBM model to predict the future AQI for all horizons."""
    feats = add_time_features(features)
    X = pd.DataFrame([feats]).reindex(columns=LGB_FEATURES, fill_value=0)
    preds = lgb_model.predict(X)[0]
    return {f"{h}h": int(preds[i]) for i, h in enumerate(lgb_horizons)}

# Step 1 : loop through each location
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
        base_pollutants = get_realtime_pollutants(loc["lat"], loc["lon"])
        feat = {"latitude": loc["lat"], "longitude": loc["lon"], **base_pollutants}
        aqi_now = rf_predict_now(feat)
        forecast = lgb_predict_future(feat)
        
        results.append({
            "id": loc["id"],
            "full_name": loc["full_name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "aqi_now": aqi_now,
            "forecast": forecast,
            "pollutants": base_pollutants
        })
        time.sleep(0.1)

    # Save the new results and the current time to the cache
    CACHE["data"] = results
    CACHE["timestamp"] = datetime.now()
    
    return results

def build_map_for_card():
    """Builds a static Folium map for the landing page card."""
    all_preds = predict_all_locations()
    if not all_preds:
        return ""
    
    center_lat = float(np.mean([p["lat"] for p in all_preds]))
    center_lon = float(np.mean([p["lon"] for p in all_preds]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")
    for p in all_preds:
        CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=16,
            color=aqi_color(p["aqi_now"]),
            fill=True,
            fill_color=aqi_color(p["aqi_now"]),
            fill_opacity=0.6,  # Changed this value for semi-transparency
            tooltip=p["full_name"]
        ).add_to(m)
    return m._repr_html_()

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
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=9, color=col, fill=True, fill_color=col, fill_opacity=0.9,
            tooltip=tooltip_text, popup=tooltip_text
        ).add_to(m)
        
    time_slices, time_labels, now = [], [], datetime.now()
    for h in lgb_horizons:
        slice_points = []
        for p in all_preds:
            aqi_val = p["forecast"].get(f"{h}h")
            if aqi_val:
                slice_points.append([p["lat"], p["lon"], aqi_val])
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
    """Renders the new front page."""
    return render_template("front_page.html")

@app.route("/dashboard")
def dashboard():
    """Renders the two-card dashboard page."""
    all_preds = predict_all_locations()
    air_card_map = build_map_for_card()
    
    initial_data = all_preds[0] if all_preds else None
    
    return render_template(
        "index.html",
        air_map_html=air_card_map,
        locations_data=json.dumps(all_preds),
        initial_air_data=json.dumps(initial_data),
    )
    
@app.route("/air_dashboard")
def air_dashboard():
    """Full air quality dashboard page."""
    all_preds = predict_all_locations()
    folium_html = build_full_dashboard_map()
    
    default_loc = all_preds[0] if all_preds else None
    
    if default_loc is None:
        chart_series = []
        default_location_name = "No Locations"
    else:
        chart_series = build_plotly_series(default_loc)
        default_location_name = default_loc["full_name"]

    return render_template(
        "air_dashboard.html",
        map_html=folium_html,
        locations=all_preds,
        default_location_name=default_location_name,
        chart_series=json.dumps(chart_series)
    )

@app.route("/api/forecast_series")
def api_forecast_series():
    """API endpoint to fetch forecast series for a specific location."""
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
    """Placeholder for the water quality dashboard."""
    return "<h1>Water Dashboard</h1><p>work in progress!</p>"


if __name__ == "__main__":
    app.run(debug=True)