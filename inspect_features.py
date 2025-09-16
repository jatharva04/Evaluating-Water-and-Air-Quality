import joblib

# Path to your saved feature list
FEATURE_PATH = "models/wqi_forecast_features.pkl"

# Load the list from the file
feature_list = joblib.load(FEATURE_PATH)

print(f"The model was trained on {len(feature_list)} features:")
print(feature_list)