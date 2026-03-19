from services.ml_service import *
from services.chatbot_service import *

print("---- UNIT TESTING START ----")

# UT01 - Water data
try:
    water = get_all_water_data()
    print("UT01 PASS" if water else "UT01 FAIL")
except:
    print("UT01 FAIL")

# UT02 - AQI
try:
    res = get_aqi_for_city("New Panvel", LOCATIONS, aqi_model)
    print("UT02 PASS" if isinstance(res, dict) else "UT02 FAIL")
except:
    print("UT02 FAIL")

# UT03 - Pollutant
try:
    res = get_pollutant_for_city("Thane", "pm2.5", LOCATIONS_CHATBOT)
    print("UT03 PASS" if isinstance(res, str) else "UT03 FAIL")
except:
    print("UT03 FAIL")

# UT04 - Chatbot
try:
    res = get_chatbot_response("What is AQI in Thane?", "get_aqi", [], 1.0)
    print("UT04 PASS" if isinstance(res, dict) else "UT04 FAIL")
except:
    print("UT04 FAIL")

print("---- UNIT TESTING END ----")