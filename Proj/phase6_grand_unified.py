import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import json

# --- CONFIGURATION ---
MODEL_FILE = 'epilim_brain.json'
ENCODER_FILE = 'disease_encoder.npy'
INFRA_FILE = 'MASTER_Analytical_Base_Table.csv'

print("--- PHASE 6: GRAND UNIFIED RISK ENGINE (ACTIVATED) ---")

# 1. LOAD THE INTEGRATED SYSTEMS
print("Loading Neural Brain & Infrastructure Grid...")
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)

# Load Disease Decoder
encoder_classes = np.load(ENCODER_FILE, allow_pickle=True)
le = LabelEncoder()
le.classes_ = encoder_classes

# Load Static Infrastructure Data (The "Ground Truth")
df_infra = pd.read_csv(INFRA_FILE)

# 2. THE INTELLIGENT BRIDGE (Fuzzy Matcher)
# Connects the XGBoost predictions (District A) to the Infrastructure Database (District A)
infra_districts = df_infra['district'].unique()

def get_district_capacity(search_name):
    # Find best match in Infrastructure DB
    match, score = process.extractOne(search_name, infra_districts)
    if score < 80:
        return None # No reliable data found
    
    # Get the row
    row = df_infra[df_infra['district'] == match].iloc[0]
    
    # Extract Key Metrics safely (using keyword search again to be safe)
    def get_val(keywords):
        for col in df_infra.columns:
            if all(k in col for k in keywords): return row[col]
        return 0

    return {
        "matched_name": match,
        "phc_count": get_val(['phc', 'total']),
        "chc_count": get_val(['chc', 'total']),
        "sanitation_coverage": get_val(['sanitation']),
        "stunting_rate": get_val(['stunted'])
    }

# 3. THE "COLLAPSE" CALCULATOR
# This is the logic that finds the "Unseen" Cascading Risk
def analyze_scenario(dist_name, lat, lon, month, rain, temp, lai, disease_name):
    # A. PREDICT THE OUTBREAK (Demand)
    try:
        d_code = le.transform([disease_name])[0]
    except:
        return {"error": "Disease not in training set"}

    # Inputs: ['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code']
    # We simulate 'prev_rain' as current rain for worst-case estimation
    input_vec = pd.DataFrame([[month, lat, lon, rain, temp, lai, rain, temp, d_code]], 
                             columns=['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code'])
    
    pred_cases = max(0, int(model.predict(input_vec)[0]))

    # B. CHECK CAPACITY (Supply)
    infra = get_district_capacity(dist_name)
    if not infra:
        return {"district": dist_name, "error": "No infrastructure data found"}

    # C. CALCULATE CASCADING RISK (The "Hidden Pattern")
    # Rule 1: Healthcare Load -> If Cases > (PHCs * 50), the system crashes.
    hospital_capacity = (infra['phc_count'] * 2) + (infra['chc_count'] * 15) # Isolation bed capacity
    if hospital_capacity == 0: hospital_capacity = 100 # Avoid divide by zero
    
    load_ratio = pred_cases / hospital_capacity
    
    # Rule 2: Sanitation Multiplier -> If sanitation is bad, risk is stickier/harder to control
    sanitation_risk = 1.0
    if infra['sanitation_coverage'] < 50: sanitation_risk = 1.5 # 50% harder to contain
    
    final_risk_score = load_ratio * sanitation_risk

    # D. GENERATE INTELLIGENCE REPORT
    status = "STABLE"
    action = "Routine Surveillance"
    
    if final_risk_score > 2.0:
        status = "SYSTEM COLLAPSE"
        action = "🚨 EMERGENCY: Patient load exceeds hospital capacity by 200%. Deploy Field Hospitals."
    elif final_risk_score > 1.0:
        status = "CRITICAL STRESS"
        action = "⚠️ URGENT: Hospitals at max capacity. Divert patients to neighboring districts."
    elif final_risk_score > 0.5:
        status = "HIGH LOAD"
        action = "Prepare overflow wards."

    return {
        "district": infra['matched_name'],
        "scenario": f"{disease_name} Outbreak ({rain}mm Rain)",
        "prediction": {
            "cases": pred_cases,
            "drivers": f"Weather: {temp}C, Rain: {rain}mm"
        },
        "capacity": {
            "hospitals": int(infra['phc_count'] + infra['chc_count']),
            "est_bed_capacity": int(hospital_capacity),
            "sanitation_level": f"{infra['sanitation_coverage']:.1f}%"
        },
        "analysis": {
            "risk_status": status,
            "collapse_probability": f"{min(final_risk_score*20, 99):.1f}%",
            "reasoning": f"Predicted {pred_cases} cases vs Capacity for {int(hospital_capacity)}. Sanitation multiplier: {sanitation_risk}x."
        },
        "recommendation": action
    }

# 4. RUN "REAL WORLD" SIMULATIONS
print("\n" + "="*60)
print("   CASCADING RISK ONTOLOGY ENGINE - LIVE SIMULATION")
print("="*60 + "\n")

scenarios = [
    # High Rain in Dehradun (High Capacity, Good Sanitation)
    ("Dehradun", 30.3, 78.0, 8, 300.0, 26.0, 4.0, "Malaria"),
    
    # High Rain in Kanpur (High Capacity, Mixed Sanitation)
    ("Kanpur Nagar", 26.4, 80.3, 8, 300.0, 32.0, 2.5, "Dengue"),
    
    # High Rain in a Vulnerable District (Leh/Ladakh - Low Capacity)
    ("Leh", 34.1, 77.5, 7, 50.0, 18.0, 0.5, "Acute Diarrhoeal Disease"),
    
    # High Rain in Patna (High Density, Stress Test)
    ("Patna", 25.6, 85.1, 7, 400.0, 30.0, 3.0, "Cholera")
]

results = []
for s in scenarios:
    res = analyze_scenario(*s)
    results.append(res)
    
    # Pretty Print for Terminal
    print(f">> DISTRICT: {res['district']}")
    print(f"   SCENARIO: {res['scenario']}")
    print(f"   PREDICTION: {res['prediction']['cases']} Cases")
    print(f"   INFRASTRUCTURE: {res['capacity']['hospitals']} Clinics ({res['capacity']['sanitation_level']} Sanitation)")
    print(f"   RISK ASSESSMENT: {res['analysis']['risk_status']}")
    print(f"   LOGIC: {res['analysis']['reasoning']}")
    print(f"   ACTION: {res['recommendation']}")
    print("-" * 60)

# 5. EXPORT FINAL JSON FOR FRONTEND
with open('grand_unified_output.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n[SYSTEM] Full Analysis exported to 'grand_unified_output.json'. Ready for Dashboard.")