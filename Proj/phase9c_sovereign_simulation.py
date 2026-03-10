import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import json
import math

MODEL_FILE = 'sovereign_brain_perfected.json'
ENCODER_FILE = 'disease_encoder_v2.npy'
POPULATION_FILE = 'System_Collapse_Baseline.csv'
WATER_FILE = 'Cleaned_Historical_Water_Quality.csv'
VAX_FILE = 'Cleaned_Immunization_History.csv'

print("--- PHASE 9c: TRUE EXPLAINABLE AI ENGINE ---")

model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
le = LabelEncoder()
le.classes_ = np.load(ENCODER_FILE, allow_pickle=True)

df_pop = pd.read_csv(POPULATION_FILE)
df_water = pd.read_csv(WATER_FILE)
df_vax = pd.read_csv(VAX_FILE)

df_pop.columns = [c.strip().lower() for c in df_pop.columns]
df_water.columns = [c.strip().lower() for c in df_water.columns]
df_vax.columns = [c.strip().lower() for c in df_vax.columns]

all_districts = df_pop['district'].unique()

def safe_float(val, fallback):
    if pd.isna(val) or math.isnan(val): return fallback
    return float(val)

def get_district_intel(search_name):
    match, score = process.extractOne(search_name, all_districts)
    if score < 80: return None
    row_pop = df_pop[df_pop['district'] == match].iloc[0]
    row_water = df_water[df_water['district'] == match].mean(numeric_only=True)
    row_vax = df_vax[df_vax['district'] == match]
    
    vax_rate = safe_float(row_vax.iloc[0].get('fully_vaccinated_%') if not row_vax.empty else None, 70.0)
    return {
        "name": match,
        "population": int(safe_float(row_pop.get('total_population'), 10000)),
        "beds": max(1, int(safe_float(row_pop.get('estimated_total_beds'), 10))),
        "bod": safe_float(row_water.get('b.o.d. (mg/l)'), 1.0),
        "fecal": safe_float(row_water.get('fecal coliform (mpn/100ml)'), 10.0),
        "tds": safe_float(row_water.get('tds_level'), 150.0),
        "vax_rate": vax_rate
    }

def run_simulation(dist_name, month, rain, temp, lai, disease_name):
    intel = get_district_intel(dist_name)
    if not intel: return {"error": "Not found"}
    if disease_name not in le.classes_: return {"error": "Unknown disease"}

    d_code = le.transform([disease_name])[0]
    
    # Feature list MUST match the new training script exactly
    features = ['month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp', 'bod', 'fecal_coliform', 'tds', 'vax_full', 'vax_measles', 'disease_code']
    input_vec = pd.DataFrame([[month, rain, temp, lai, rain, temp, intel['bod'], intel['fecal'], intel['tds'], intel['vax_rate'], intel['vax_rate'], d_code]], columns=features)
    
    predicted_cases = max(0, int(model.predict(input_vec)[0]))
    predicted_cases = min(predicted_cases, intel['population']) 

    # ====================================================================
    # ML FIX: TRUE EXPLAINABILITY (XGBOOST PREDICTION CONTRIBUTIONS)
    # ====================================================================
    # This asks the model: "For this specific calculation, what drove the number up?"
    dmatrix = xgb.DMatrix(input_vec)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)[0]
    
    # Pair features with their exact mathematical contribution (ignoring the final bias term)
    feature_contributions = list(zip(features, contribs[:-1]))
    
    # Sort by absolute impact (what pushed the dial the most)
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Make the feature names readable for the government
    readable_map = {'preci': 'Heavy Rainfall', 'fecal_coliform': 'Water Toxicity (Fecal)', 'vax_full': 'Low Vaccination Shield', 'temp_celsius': 'Temperature Spikes', 'lai': 'Vegetation/Standing Water'}
    top_feature_raw = feature_contributions[0][0]
    # Skip disease_code and month as "primary reasons" because they are contextual, get the next environmental trigger
    for f, val in feature_contributions:
        if f not in ['disease_code', 'month']:
            top_feature_raw = f
            break
            
    primary_driver = readable_map.get(top_feature_raw, top_feature_raw)

    # Calculate System Load
    infection_rate = (predicted_cases / intel['population']) * 10000
    hospital_demand = int(predicted_cases * 0.20)
    bed_deficit = hospital_demand - intel['beds']
    
    status = "STABLE"
    if bed_deficit > 0: status = "SYSTEM COLLAPSE"
    elif hospital_demand > (intel['beds'] * 0.8): status = "CRITICAL STRESS"

    people_per_bed = intel['population'] / intel['beds']
    ai_governance_insight = "✅ INFRASTRUCTURE ADEQUATE"
    if people_per_bed > 1500:
        ai_governance_insight = f"⚠️ DECEPTIVE SURPLUS: 1 bed is forced to serve {int(people_per_bed)} people (WHO Limit: 333). Chronic baseline overload detected."

    print(f"\n{'='*60}")
    print(f"🏛️  DISTRICT: {intel['name'].upper()}")
    print(f"🔮 PREDICTION: {predicted_cases} {disease_name} Cases")
    print(f"🧠 AI X-RAY: Triggered primarily by >> {primary_driver.upper()} <<")
    print(f"🛑 STATUS: {status}")
    print(f"💡 INSIGHT: {ai_governance_insight}")
    print(f"{'='*60}")

# RUN SIMULATIONS
print("\n[SYSTEM] Executing Deep X-Ray Simulations...")
run_simulation("Kanpur Nagar", 8, 300.0, 32.0, 2.5, "Dengue")
run_simulation("Dehradun", 8, 250.0, 26.0, 4.0, "Malaria")
run_simulation("Leh", 7, 50.0, 18.0, 0.5, "Acute Diarrhoeal Disease")
run_simulation("Aizawl", 6, 200.0, 28.0, 3.0, "Malaria")