import requests
import json

# --- CONFIGURATION ---
# ✅ Replace with your actual Google Cloud API Key
API_KEY = 'AIzaSyB5niPLqZvk1i0qUjyJ094_phKx1_JbBbM'

# ✅ Change these to the coordinates you want to test
LATITUDE = 19.0734
LONGITUDE = 72.9972

# --- SCRIPT LOGIC ---
def get_google_aqi(lat, lon, api_key):
    """
    Fetches and prints current AQI data from the Google Air Quality API.
    """
    if not api_key or api_key == 'YOUR_GOOGLE_API_KEY':
        print("❌ ERROR: Please replace 'YOUR_GOOGLE_API_KEY' with a valid key.")
        print("➡️  You need a Google Cloud Project with the Air Quality API and Billing enabled.")
        return

    api_url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
    
    # The request payload is sent as JSON, not URL parameters
    payload = {
        "location": {
            "latitude": lat,
            "longitude": lon
        }
    }
    
    # The API key is sent as a query parameter in the URL
    params = {'key': api_key}
    
    print(f"🔍 Fetching Google AQI data for Latitude: {lat}, Longitude: {lon}...")

    try:
        response = requests.post(api_url, params=params, json=payload)
        response.raise_for_status()
        data = response.json()

        if data and 'indexes' in data and data['indexes']:
            # Google's API provides data for different standards (e.g., US EPA, India CPCB)
            # We'll use the universal AQI (uAQI) for a general comparison
            uaqi_data = next((idx for idx in data['indexes'] if idx['code'] == 'uaqi'), None)
            
            if not uaqi_data:
                print("❌ ERROR: Universal AQI (uAQI) data not found in the response.")
                return

            print("\n--- ✅ Google API Response Successful ---")
            print(f"\nGoogle's Universal AQI (uAQI): {uaqi_data['aqiDisplay']}")
            print(f"Category: {uaqi_data['category']}")
            print(f"Dominant Pollutant: {uaqi_data['dominantPollutant'].upper()}")
            print(f"Color Code: {uaqi_data['color']}")

            print("\nHealth Recommendations:")
            print(f" • General Population: {data['healthRecommendations']['generalPopulation']}")
            print(f" • Lung Disease Population: {data['healthRecommendations']['lungDiseasePopulation']}")
            
            print("\n------------------------------------")
        else:
            print("❌ ERROR: The API response was empty or did not contain AQI data.")
            print("Full Response:", data)

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: An API error occurred: {e}")
        if e.response:
             print("Error Details:", e.response.text)


# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    get_google_aqi(LATITUDE, LONGITUDE, API_KEY)