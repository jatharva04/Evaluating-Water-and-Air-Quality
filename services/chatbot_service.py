from config import *
from services.ml_service import *
import spacy
from thefuzz import process, fuzz

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
    all_pollutants, _, _ = get_realtime_pollutants(location_info["lat"], location_info["lon"])
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
    api_key = os.getenv("OPENWEATHER_API_KEY")
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"

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


def get_recommendation_for_aqi_value(value):
    """Returns a specific recommendation based on a numerical AQI value."""
    value = int(value)
    if value <= 50:
        return f"With an AQI of {value}, the air is 'Good'. It's a great day for outdoor activities!"
    elif value <= 100:
        return f"With an AQI of {value}, the air is 'Satisfactory'. It's generally safe to be outside."
    elif value <= 200:
        return f"With an AQI of {value}, the air is 'Moderate'. Sensitive groups, like children and the elderly, should reduce prolonged outdoor exertion."
    else: # value > 200
        return f"An AQI of {value} is 'Poor' or worse. It's strongly recommended to avoid outdoor activities, especially for sensitive groups."

# You will also need a helper to format the final AQI string
def get_aqi_response_string(city_name):
    """Formats the dictionary returned by get_aqi_for_city into a human-readable string."""
    aqi_data = get_aqi_for_city(city_name, LOCATIONS_CHATBOT, aqi_model)
    if isinstance(aqi_data, dict):
        return (f"The current predicted AQI for **{aqi_data['city']}** is **{int(aqi_data['aqi_500_scale'])}** "
                f"on the official 0-500 scale (Level **{aqi_data['aqi_1_to_5_scale']}** on the dashboard), "
                f"which is considered **'{aqi_data['category']}'**.")
    else:
        return aqi_data
        
def get_wqi_response_string(city_name):
    """Formats the live Machine Learning WQI prediction for a city into a string."""
    matched_city = find_best_location_match(city_name, LOCATIONS_CHATBOT)
    if not matched_city:
        return f"I'm sorry, I couldn't find water quality data for '{city_name}'."
        
    all_water_data = get_all_water_data()
    water_info = next((
        loc for loc in all_water_data 
        if matched_city.lower() in loc["full_name"].lower().strip()
    ), None)
    if water_info:
        return (f"Currently, the predicted ML Water Quality Index (WQI) for **{water_info['full_name']}** is **{water_info['wqi']:.0f}/100**.\n\n"
                f"This classifies the location as **'{water_info['classification']}'** "
                f"*(Remark: {water_info['remark']})*.")
    else:
        return f"Sorry, I couldn't generate a live Water Quality prediction for {matched_city.capitalize()} at the moment."

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
            
    if state == "waiting_for_city_for_wqi":
        city_name = user_input.strip()
        matched_city = find_best_location_match(city_name, LOCATIONS_CHATBOT)
        if matched_city:
            response_text = get_wqi_response_string(matched_city) 
            return {
                "response": response_text, 
                "session_data": {"state": None, "last_topic": "water"}
            }
        else:
            return {
                "response": f"I couldn't find a supported location matching '{city_name}'. Please try another location.",
                "session_data": session_data
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
    elif predicted_intent in ['get_wqi', 'explain_wqi']:
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
                
                # Return early to prevent the NoneType error
                return {"response": response_text, "session_data": session_data}

            user_city_query = city_entity[0].strip()
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
                city_name = city_entity[0].strip()
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

        elif predicted_intent == 'get_wqi':
            session_data['last_topic'] = 'water'
            city_entity = find_entity(extracted_entities, 'city')
            
            if not city_entity:
                if any(word in user_input.lower() for word in ["here", "near me"]):
                    city_name = "Ambernath" # Same default behavior as AQI
                else:
                    response_text = "For which city would you like to know the Water Quality Index (WQI)?"
                    session_data['state'] = "waiting_for_city_for_wqi"
                    return {"response": response_text, "session_data": session_data}
            else:
                city_name = city_entity[0].strip()
            
            # Use the existing function that wraps all of this nicely
            response_text = get_wqi_response_string(city_name)
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
                        # Get the real Machine Learning WQI score from the live cache / CSV dataset
                        all_water_data = get_all_water_data()
                        water_info = next((loc for loc in all_water_data if loc["full_name"].lower() == matched_city.lower()), None)
                        
                        # Fallback to the hardcoded dictionary only if the location isn't in water_rep_data.csv yet
                        wqi_score = water_info["wqi"] if water_info else REPRESENTATIVE_WQI.get(matched_city, 0)
                        
                        # Convert 0-500 AQI (lower is better) to a 0-100 score (higher is better)
                        aqi_500_value = aqi_data['aqi_500_scale']
                        air_score = max(0, 100 - (aqi_500_value / 500) * 100)
                        
                        # Average the two 0-100 scores for a final livability score
                        livability_score = (air_score + wqi_score) / 2
                        
                        results.append({
                            'city': matched_city,
                            'aqi_500': aqi_data['aqi_500_scale'],
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
                "Here is the detailed breakdown (ranked by overall Livability Score):<br>"
            )

            for res in results:
                response_text += f"- **{res['city']}**: Livability Score: {res['score']:.0f}/100 <i>(AQI: {res['aqi_500']:.0f}, WQI: {res['wqi']:.0f})</i><br>"
            
            response_text += "<br><span style='font-size: 0.9em; color: gray;'>*Note: For the overall Livability Score and WQI, higher is better. For standard AQI, lower is better.</span>"
            return {"response": response_text, "session_data": session_data}

    return {"response": response_text, "session_data": session_data}
# ---------------------------------------- END OF CHATBOT ---------------------------------------

