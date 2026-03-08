from flask import Blueprint, request, jsonify
import pandas as pd
from datetime import datetime
from config import *
from services.ml_service import *
from services.chatbot_service import *
from utils.map_helpers import *

api_bp = Blueprint('api', __name__)

@api_bp.route("/api/predict_wqi", methods=["POST"])
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

@api_bp.route("/api/chat", methods=["POST"])
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

    # --- Debug Prints for Terminal ---
    print("\n--- CHATBOT DEBUG ---")
    print(f"User Input: '{user_input}'")
    print(f"--> Predicted Intent: {predicted_intent} (Score: {score:.2f})")
    if extracted_entities:
        print(f"--> Extracted Entities: {extracted_entities}")
    print("---------------------\n")
    
    # Pass the NLP results to your main logic function
    response_data = get_chatbot_response(
        user_input,
        predicted_intent,
        extracted_entities,
        score,
        session_data
    )

    return jsonify(response_data)

@api_bp.route("/api/forecast_series")
def api_forecast_series():
    id = request.args.get("id", "").strip()
    if not id: 
        return jsonify({"ok": False, "error": "Missing 'id'"}), 400
    all_preds = predict_all_locations()
    loc = next((p for p in all_preds if str(p["id"]) == id), None)
    if not loc: 
        return jsonify({"ok": False, "error": "Location not found"}), 404
    return jsonify({"ok": True, "name": loc['full_name'], "series": build_plotly_series(loc)})


@api_bp.route("/api/location_details")
def api_location_details():
    id = request.args.get("id", "").strip()
    if not id:
        return jsonify({"ok": False, "error": "Missing 'id'"}), 400
        
    loc = get_location_ml_details(id)
    if not loc:
        return jsonify({"ok": False, "error": "Location not found"}), 404
        
    return jsonify({
        "ok": True,
        "recommendations": loc.get("recommendations"),
        "lime_explanation": loc.get("lime_explanation"),
        "shap_explanation": loc.get("shap_explanation")
    })

@api_bp.route("/api/wqi_forecast_series")
def api_wqi_forecast_series():
    """API endpoint to fetch WQI forecast series for a specific location."""
    id_str = request.args.get("id", "").strip()
    if not id_str: 
        return jsonify({"ok": False, "error": "Missing 'id'"}), 400
        
    # Find forecast data from the dynamically updated cache
    all_forecasts = get_all_water_forecasts()
    loc = next((p for p in all_forecasts if str(p["id"]) == id_str), None)
    
    if not loc: 
        return jsonify({"ok": False, "error": "Location not found"}), 404
        
    return jsonify({"ok": True, "name": loc['full_name'], "series": build_wqi_plotly_series(loc['forecast'])})

@api_bp.route("/api/wqi_location_details")
def api_wqi_location_details():
    """API endpoint for WQI recommendations and LIME."""
    id_str = request.args.get("id", "").strip()
    if not id_str:
        return jsonify({"ok": False, "error": "Missing 'id'"})
        
    # Find location data using the dynamically lazy-loading ML helper
    loc = get_water_location_ml_details(id_str)
    
    if not loc:
        return jsonify({"ok": False, "error": "Location not found"})
        
    return jsonify({
        "ok": True,
        "recommendation": loc.get("recommendation"),
        "lime_explanation": loc.get("lime_explanation"),
        "shap_explanation": loc.get("shap_explanation")
    })

