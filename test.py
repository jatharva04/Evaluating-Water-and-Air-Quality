import requests
import json

# --- CONFIGURATION ---
# ✅ Replace with your actual API Key
API_KEY = '6a10ef7f599b4187e0cd396738b9fa41' 

# ✅ Change these to any coordinates you want to test
LATITUDE = 19.0734 
LONGITUDE = 72.9972

# --- SCRIPT LOGIC ---
def get_current_aqi(lat, lon, api_key):
    """
    Fetches and prints the current AQI data from OpenWeather for a single location.
    """
    # Check if the API key has been replaced
    if not api_key or api_key == 'YOUR_API_KEY':
        print("❌ ERROR: Please replace 'YOUR_API_KEY' with your actual OpenWeather API key.")
        return

    # Construct the API request URL and parameters
    api_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }

    print(f"🔍 Fetching data for Latitude: {lat}, Longitude: {lon}...")

    try:
        # Make the API call
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)

        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected data
        if data and 'list' in data and data['list']:
            aqi_data = data['list'][0]
            
            # Print the results in a readable format
            print("\n--- ✅ API Response Successful ---")
            
            print(f"\nOpenWeather's AQI Assessment: {aqi_data['main']['aqi']}")
            
            print("\nPollutant Components (in µg/m³):")
            # Use json.dumps for pretty printing the components dictionary
            print(json.dumps(aqi_data['components'], indent=4))
            
            print("\n------------------------------------")
        else:
            print("❌ ERROR: The API response was empty or malformed.")

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: An error occurred while contacting the API: {e}")

# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    get_current_aqi(LATITUDE, LONGITUDE, API_KEY)