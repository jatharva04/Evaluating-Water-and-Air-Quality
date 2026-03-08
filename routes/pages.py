from flask import Blueprint, render_template
import json
from config import *
from services.ml_service import *
from utils.map_helpers import *

pages_bp = Blueprint('pages', __name__)

@pages_bp.route("/")
def landing_page(): 
    return render_template("front_page.html")

@pages_bp.route("/dashboard")
def dashboard():
    # --- Get Air Data ---
    all_air_preds = predict_all_locations() or []
    air_card_map = build_generic_map(all_air_preds, 'aqi_now', aqi_color)
    initial_air_data = all_air_preds[0] if all_air_preds else None
    if initial_air_data:
        initial_air_data = get_location_ml_details(initial_air_data["id"]) or initial_air_data

    # --- Get Water Data ---
    all_water_data = get_all_water_data() or []
    water_card_map = build_generic_map(all_water_data, 'wqi', wqi_color)
    initial_water_data = all_water_data[0] if all_water_data else None
    if initial_water_data:
        initial_water_data = get_water_location_ml_details(initial_water_data["id"]) or initial_water_data

    # Append timestamp strings from cache if available
    air_updated_time = CACHE["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if CACHE["timestamp"] else "Unknown"
    water_updated_time = CACHE_WATER["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if CACHE_WATER["timestamp"] else "Unknown"

    return render_template(
        "index.html",
        # Air Data
        air_map_html=air_card_map,
        air_locations_data=all_air_preds,
        initial_air_data=initial_air_data,
        air_updated_time=air_updated_time,
        # Water Data
        water_map_html=water_card_map,
        water_locations_data=all_water_data,
        initial_water_data=initial_water_data,
        water_updated_time=water_updated_time
    )

@pages_bp.route("/air_dashboard")
def air_dashboard():
    all_preds = predict_all_locations()
    folium_html = build_full_dashboard_map()
    default_loc = all_preds[0] if all_preds else None
    if default_loc is None:
        chart_series, default_location_name = [], "No Locations"
    else:
        chart_series = build_plotly_series(default_loc)
        default_location_name = default_loc["full_name"]
        default_loc = get_location_ml_details(default_loc["id"]) or default_loc
        
    air_updated_time = CACHE["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if CACHE["timestamp"] else "Unknown"
        
    return render_template(
        "air_dashboard.html", map_html=folium_html, locations=all_preds,
        default_location_name=default_location_name, chart_series=json.dumps(chart_series),
        initial_details=default_loc, air_updated_time=air_updated_time
    )

@pages_bp.route("/water_dashboard")
def water_dashboard(): 
    """Full water quality dashboard page with forecast."""
    # Fetch dynamically from the fresh cache
    all_water_data = get_all_water_data()
    all_forecasts = get_all_water_forecasts()
    water_forecast_dict = {str(loc.get("id")): loc for loc in all_forecasts}
    # Attach forecast to each location using its ID (robust)
    for loc in all_water_data:
        forecast_data = water_forecast_dict.get(str(loc.get("id")))
        loc["forecast"] = forecast_data.get("forecast", {}) if forecast_data else {}

    folium_html = build_full_water_dashboard_map(all_water_data)
    default_loc = all_water_data[0] if all_water_data else None
    if default_loc is None:
        chart_series, default_location_name = [], "No Locations"
    else:
        forecast_loc = water_forecast_dict.get(str(default_loc["id"]))
        chart_series = build_wqi_plotly_series(forecast_loc["forecast"]) if forecast_loc else []
        default_location_name = default_loc["full_name"]
        default_loc = get_water_location_ml_details(default_loc["id"]) or default_loc
        
    water_updated_time = CACHE_WATER["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if CACHE_WATER["timestamp"] else "Unknown"

    return render_template(
        "water_dashboard.html", 
        map_html=folium_html, 
        locations=all_water_data,
        default_location_name=default_location_name, 
        chart_series=chart_series,
        initial_details=default_loc,
        water_updated_time=water_updated_time
    )

