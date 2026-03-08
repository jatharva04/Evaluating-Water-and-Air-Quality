# 🌱 AI-Powered Environmental Dashboard

A **comprehensive environmental monitoring platform** that leverages cutting-edge machine learning to provide **real-time insights** and **predictive forecasts** for both Air and Water quality in the Mumbai-Thane region.

---

## 🚀 Key Features

### 📊 Air Quality Dashboard
- **Live AQI Prediction**: Uses a **Random Forest model** to predict the current Air Quality Index (AQI) based on real-time pollutant data from OpenWeather API.
- **72-Hour Forecast**: A **LightGBM time-series model** visualizes future air quality trends with animated heatmaps.
- **Explainable AI (XAI)**: Integrated **SHAP** and **LIME** insights to show exactly which pollutants (PM2.5, NO2, etc.) are driving the current AQI.

### 💧 Water Quality Dashboard  
- **Real-Time WQI Analysis**: Employs an **XGBoost regression model** to calculate the Water Quality Index (WQI) across multiple stations using the latest MPCB baseline data.
- **6-Month Predictive Forecast**: Forecasts seasonal water quality trends through Random Forest Regressor.
- **Detailed Recommendations**: Provides actionable health and safety recommendations based on water classification.

### 🤖 Intelligent Chatbot Assistant
- **NLU-Powered Core**: Uses trained NER (Named Entity Recognition) and Textcat models to understand user intent.
- **Livability Comparisons**: Compare two locations (e.g., "Ambernath vs Badlapur") to receive a **Livability Score** (weighted average of Air/Water quality).
- **Interactive Knowledge Base**: Get instant definitions for pollutants (PM10, BOD, COD) and AQI/WQI categories.
- **Premium UX**: Smooth auto-scrolling interface with glassmorphic design and Escape-key closure.

### 🛡️ Data Transparency & Freshness
- **"Last Updated" Badges**: Every dashboard clearly displays exactly when data was last fetched from the backend sensors or CSV baselines.
- **Verified Datasets**: The water quality baseline is consistently updated with the latest reports from MPCB (currently featuring **December 2025** data).

---

## 🛠️ Technology Stack

**Backend & Integration:**
- **Python / Flask**: Robust backend management.
- **Spacy / TheFuzz**: Advanced Natural Language Processing for the chatbot.

**Machine Learning Ecosystem:**
- **Scikit-Learn**: Core modeling (Random Forest).
- **XGBoost & LightGBM**: High-performance gradient boosting for forecasts.
- **SHAP & LIME**: Model interpretability.

**Frontend & Visualization:**
- **Folium & Plotly**: Dynamic mapping and interactive charting.
- **Vanilla CSS**: Premium "Glassmorphism" UI design.
- **Luxon.js**: Timezone-aware timestamping.

---

## 📁 Project Structure

- `services/`: Core logic for ML predictions and Chatbot processing.
- `routes/`: Flask blueprints for pages and RESTful API endpoints.
- `data/`: Curated datasets for AQI, WQI, and NLP training.
- `templates/`: Responsive HTML5 dashboards.

---

## 📌 Development Status
The project has transitioned from a proof-of-concept to a fully functional predictive platform. Future enhancements focus on:
- Integration of live automated water sensors (IoT).
- Enhanced multi-city pollutant drift analysis.
- Personalized user notification systems.
