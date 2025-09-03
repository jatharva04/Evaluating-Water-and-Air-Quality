AI-Powered Environmental Dashboard
This project is a web-based dashboard that leverages machine learning models to provide real-time predictions and 72-hour forecasts for air and water quality. Built with Python and Flask, the application offers an intuitive interface for monitoring environmental data.

Key Features
Live AQI Prediction: Uses a Random Forest model to predict the current Air Quality Index (AQI) based on real-time pollutant data.

72-Hour Air Quality Forecast: A LightGBM time-series model provides a 72-hour forecast of future air quality trends.

Dynamic Dashboard: An interactive dashboard with a Folium map visualizes current AQI markers and animated heatmaps for future forecasts.

Water Quality Monitoring: A placeholder for a future feature that will use a regression model to predict water quality.

Chatbot Assistant: A semi-advanced, rule-based chatbot helps users understand AQI categories and pollutant definitions.

Technology Stack
Backend: Python, Flask, Pandas, NumPy

Machine Learning: Scikit-learn, LightGBM

Data Visualization: Folium, Chart.js, Luxon.js
