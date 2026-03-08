import folium
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from folium import Map, CircleMarker, LayerControl
from folium.plugins import HeatMapWithTime
from branca.element import Element
from config import *
from services.ml_service import *

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

import json

def inject_post_message_listener(m: Map, map_data: list = None):
    """Injects JS to listen for parent postMessages so frontend dropdowns can highlight markers."""
    map_id = m.get_name().replace('-', '_')
    locations_json = "[]"
    if map_data:
        simplified_data = [{"lat": float(d["lat"]), "lon": float(d["lon"]), "id": d["id"]} for d in map_data]
        locations_json = json.dumps(simplified_data)

    js = """
    <script>
    var locationsData = %s;
    window.addEventListener('message', function(e) {
        if (e.data && e.data.lat && e.data.lon) {
            var theMap = window.%s;
            if (theMap) {
                if (window._highlight) { window._highlight.remove(); }
                var pinIcon = L.icon({
                    iconUrl: 'https://cdn.rawgit.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                });
                window._highlight = L.marker([e.data.lat, e.data.lon], {icon: pinIcon}).addTo(theMap);
            }
        }
    });
    
    var initInterval = setInterval(function() {
        var theMap = window.%s;
        if (theMap && locationsData.length > 0) {
            clearInterval(initInterval);
            theMap.eachLayer(function(layer) {
                if (layer instanceof L.CircleMarker) {
                    layer.on('click', function(e) {
                        var lat = e.latlng.lat;
                        var lon = e.latlng.lng;
                        var closestLoc = null;
                        var minDistance = Number.MAX_VALUE;
                        locationsData.forEach(function(loc) {
                            var dist = Math.pow(loc.lat - lat, 2) + Math.pow(loc.lon - lon, 2);
                            if (dist < minDistance) {
                                minDistance = dist;
                                closestLoc = loc;
                            }
                        });
                        if (closestLoc && minDistance < 0.0001) {
                            window.parent.postMessage({ type: 'marker_click', id: closestLoc.id }, '*');
                        }
                    });
                }
            });
        }
    }, 500);
    </script>
    """ % (locations_json, map_id, map_id)
    m.get_root().html.add_child(Element(js))
    return m


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
        label = "AQI (now)" if value_key == "aqi_now" else "WQI"
        display_val = int(value) if value_key == "aqi_now" else round(value)
        tooltip_text = f"<b>{p['full_name']}</b><br>{label}: <b>{display_val}</b>"
        CircleMarker(
            location=[p["lat"], p["lon"]], 
            radius=16, 
            color=color_function(value),
            fill=True, 
            fill_color=color_function(value), 
            fill_opacity=0.6, 
            tooltip=tooltip_text
        ).add_to(m)
    
    inject_post_message_listener(m, map_data)
    return m._repr_html_()


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
    
    inject_post_message_listener(m, all_preds)
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

def build_full_water_dashboard_map(water_data):
    """Builds the full-screen map for the water dashboard page, including a forecast heatmap."""
    if not water_data: 
        return ""
    
    center_lat = float(np.mean([p["lat"] for p in water_data]))
    center_lon = float(np.mean([p["lon"] for p in water_data]))
    m = Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

    # --- 1. Add Circle Markers for Current WQI ---
    for p in water_data:
        col = wqi_color(p["wqi"])
        tooltip_text = f"<b>{p['full_name']}</b><br>WQI (Predicted): <b>{p['wqi']:.0f}</b> ({p['classification']})"
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=9, color=col, fill=True,
            fill_color=col, fill_opacity=0.9, tooltip=tooltip_text, popup=tooltip_text
        ).add_to(m)
        
    # --- 2. Add the Animated Forecast Heatmap ---
    time_slices = []
    time_labels = []
    now = datetime.now()

    for h in WQI_HORIZONS:
        slice_points = []
        for p in water_data:
            forecast = p.get("forecast", {}) 
            wqi_val = forecast.get(f"{h}_month")
            if wqi_val: 
                slice_points.append([p["lat"], p["lon"], wqi_val])
        
        if slice_points:
            time_slices.append(slice_points)
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
    inject_post_message_listener(m, water_data)
    return m._repr_html_()

