import os
import math
import json
import random
from datetime import datetime, timedelta

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

# =========================
# 0) Setup Flask App
# =========================
app = Flask(__name__)

# =========================
# 1) Load Final Model and Data
# =========================
MODEL_PATH = os.path.join("models", "random_forest_aqi.pkl")
LGB_BUNDLE_PATH = os.path.join("models", "lightgbm_multi.pkl")
LOCATIONS_CSV = "locations.csv"

# --- Load the final chosen model ---
print("Loading final classifier model...")
final_model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Load the model for future predictions
lgb_bundle = joblib.load(LGB_BUNDLE_PATH)
lgb_model = lgb_bundle["model"]
lgb_features = lgb_bundle["features"]
lgb_horizons = lgb_bundle["horizons"]

# Required features for the models
MODEL_FEATURES = [
    "latitude", "longitude", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "dayofweek", "hour_sin", "hour_cos"
]

# --- Initialize Explainers for the Classifier ---
shap_explainer = shap.TreeExplainer(final_model)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.random.rand(100, len(MODEL_FEATURES)),
    feature_names=MODEL_FEATURES,
    class_names=['AQI 1', 'AQI 2', 'AQI 3', 'AQI 4', 'AQI 5'],
    mode='classification'
)

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
        1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22",
        4: "#e74c3c", 5: "#8e44ad"
    }.get(int(aqi_class), "#7f8c8d")

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
    """Generates varied data for predictions."""
    return {
        "co": random.uniform(50.0, 400.0), "no2": random.uniform(5.0, 60.0),
        "o3": random.uniform(10.0, 90.0), "so2": random.uniform(1.0, 20.0),
        "pm2_5": random.uniform(5.0, 75.0), "pm10": random.uniform(10.0, 100.0),
    }

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

#
# ==============================================================================
# CORRECTED FUNCTION
# ==============================================================================
#
def get_lime_explanation(features_df: pd.DataFrame) -> list:
    """Generates LIME explanation for the classifier's prediction."""
    instance_to_explain = features_df.iloc[0].values
    
    # Step 1: Tell LIME to generate explanations for ALL possible class indices (0, 1, 2, 3, 4)
    explanation = lime_explainer.explain_instance(
        instance_to_explain,
        final_model.predict_proba,
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
#
# ==============================================================================
# END OF CORRECTION
# ==============================================================================
#
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
    """Predicts AQI and generates explanations for all locations using the final classifier."""
    results = []
    for loc in LOCATIONS:
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
    return results

def lgb_predict_future(features: dict) -> dict:
    feats = add_time_features(features)
    X = pd.DataFrame([feats]).reindex(columns=lgb_features, fill_value=0)
    preds = lgb_model.predict(X)[0]
    return {f"{h}h": int(p) for h, p in zip(lgb_horizons, preds)}

def build_map_for_card():
    all_preds = predict_all_locations()
    if not all_preds: return ""
    center_lat = float(np.mean([p["lat"] for p in all_preds]))
    center_lon = float(np.mean([p["lon"] for p in all_preds]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")
    for p in all_preds:
        CircleMarker(
            location=[p["lat"], p["lon"]], radius=16, color=aqi_color(p["aqi_now"]),
            fill=True, fill_color=aqi_color(p["aqi_now"]), fill_opacity=0.6, tooltip=p["full_name"]
        ).add_to(m)
    return m._repr_html_()

def build_full_dashboard_map():
    all_preds = predict_all_locations()
    if not all_preds: return ""
    center_lat = float(np.mean([p["lat"] for p in all_preds]))
    center_lon = float(np.mean([p["lon"] for p in all_preds]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")
    for p in all_preds:
        col = aqi_color(p["aqi_now"])
        tooltip_text = f"<b>{p['full_name']}</b><br>AQI (now): <b>{p['aqi_now']}</b>"
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=9, color=col, fill=True,
            fill_color=col, fill_opacity=0.9, tooltip=tooltip_text, popup=tooltip_text
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
    if not pred_for_loc: return []
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
def landing_page(): return render_template("front_page.html")

@app.route("/dashboard")
def dashboard():
    all_preds = predict_all_locations()
    air_card_map = build_map_for_card()
    initial_data = all_preds[0] if all_preds else None
    return render_template(
        "index.html", air_map_html=air_card_map,
        locations_data=json.dumps(all_preds), initial_air_data=json.dumps(initial_data)
    )

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
    if not id: return jsonify({"ok": False, "error": "Missing 'id'"}), 400
    all_preds = predict_all_locations()
    loc = next((p for p in all_preds if str(p["id"]) == id), None)
    if not loc: return jsonify({"ok": False, "error": "Location not found"}), 404
    return jsonify({"ok": True, "name": loc['full_name'], "series": build_plotly_series(loc)})


@app.route("/water_dashboard")
def water_dashboard(): return "<h1>Water Dashboard</h1><p>work in progress!</p>"

if __name__ == "__main__":
    app.run(debug=True)
