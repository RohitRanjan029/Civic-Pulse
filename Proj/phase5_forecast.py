import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import json

print("--- PHASE 5: GENERATING GOVERNMENT FORECAST DASHBOARD ---")

# 1. LOAD THE TRAINED BRAIN
print("Loading Neural Logic...")
model = xgb.XGBRegressor()
model.load_model("epilim_brain.json")

# Load the Disease Encoder (to translate codes back to names like "Malaria")
encoder_classes = np.load('disease_encoder.npy', allow_pickle=True)
le = LabelEncoder()
le.classes_ = encoder_classes

# 2. DEFINE FORECAST SCENARIOS (The "What If")
# We create hypothetical "Next Week" weather scenarios for key districts.
# In a real app, this data would come from a Weather API.
scenarios = [
    # District, Lat, Long, Month, Rain(mm), Temp(C), LAI(Veg), Prev_Rain, Prev_Temp, Disease_Target
    ("Kanpur Nagar", 26.4, 80.3, 8, 120.5, 32.0, 2.5, 100.0, 31.0, "Dengue"),
    ("Kanpur Nagar", 26.4, 80.3, 8, 15.0, 32.0, 2.5, 10.0, 31.0, "Acute Diarrhoeal Disease"),
    ("Dehradun", 30.3, 78.0, 8, 250.0, 26.0, 4.1, 200.0, 25.0, "Malaria"),
    ("Dehradun", 30.3, 78.0, 8, 5.0, 26.0, 4.1, 0.0, 25.0, "Chickenpox"),
    ("Patna", 25.6, 85.1, 7, 300.0, 34.0, 1.8, 150.0, 33.0, "Cholera"),
    ("Leh(Ladakh)", 34.1, 77.5, 7, 2.0, 15.0, 0.5, 0.0, 14.0, "Acute Respiratory Infection")
]

# 3. RUN PREDICTIONS
results = []

print("\nRunning Predictive Engine on Future Scenarios...")

for dist, lat, lon, mon, rain, temp, lai, p_rain, p_temp, disease_name in scenarios:
    # Encode disease name to number
    try:
        d_code = le.transform([disease_name])[0]
    except:
        continue # Skip if disease wasn't in training data

    # Create input vector (Order must match training!)
    # ['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code']
    input_vector = pd.DataFrame([[mon, lat, lon, rain, temp, lai, p_rain, p_temp, d_code]], 
                                columns=['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code'])
    
    # Predict
    pred_cases = model.predict(input_vector)[0]
    pred_cases = max(0, int(pred_cases)) # No negative cases allowed
    
    # Risk Logic
    risk_level = "Low"
    if pred_cases > 50: risk_level = "Moderate"
    if pred_cases > 200: risk_level = "High"
    if pred_cases > 500: risk_level = "CRITICAL"

    # Action Logic
    action = "Routine Surveillance"
    if risk_level == "CRITICAL": action = "DEPLOY RAPID RESPONSE TEAM"
    elif risk_level == "High": action = "Increase Hospital Bed Capacity"

    # Output Object
    prediction = {
        "district": dist,
        "disease": disease_name,
        "forecast_cases": pred_cases,
        "risk_level": risk_level,
        "drivers": {
            "rain": f"{rain}mm",
            "temp": f"{temp}C"
        },
        "action_plan": action
    }
    results.append(prediction)
    
    print(f">> {dist}: Predicted {pred_cases} cases of {disease_name}. Risk: {risk_level}")

# 4. EXPORT TO JSON FOR REACT
with open('dashboard_forecast.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n[SYSTEM] Forecast generated. File 'dashboard_forecast.json' is ready for React.")