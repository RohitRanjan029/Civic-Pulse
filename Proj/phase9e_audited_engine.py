import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import math
import json

# --- CONFIGURATION ---
MODEL_FILE = 'sovereign_brain_perfected.json'
ENCODER_FILE = 'disease_encoder_v2.npy'
POPULATION_FILE = 'System_Collapse_Baseline.csv'
WATER_FILE = 'Cleaned_Historical_Water_Quality.csv'
VAX_FILE = 'Cleaned_Immunization_History.csv'
EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv' 

print("--- PHASE 9e: THE AUDITED SOVEREIGN ENGINE (FINAL) ---")

# 1. LOAD MODELS & DATA
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
le = LabelEncoder()
le.classes_ = np.load(ENCODER_FILE, allow_pickle=True)

df_pop = pd.read_csv(POPULATION_FILE)
df_water = pd.read_csv(WATER_FILE)
df_vax = pd.read_csv(VAX_FILE)
df_geo = pd.read_csv(EPICLIM_FILE)

df_pop.columns = [c.strip().lower() for c in df_pop.columns]
df_water.columns = [c.strip().lower() for c in df_water.columns]
df_vax.columns = [c.strip().lower() for c in df_vax.columns]

all_districts = df_pop['district'].unique()
districts_geo = df_geo.groupby('district').agg({'Latitude': 'first', 'Longitude': 'first'}).reset_index()

# 2. LOGISTICS DATA
RADIAL_DISTANCES = {
    "Andhra Pradesh": 17.2, "Arunachal Pradesh": 21.6, "Assam": 11.0, "Bihar": 9.9,
    "Chhattisgarh": 15.9, "Gujarat": 13.0, "Haryana": 9.9, "Himachal Pradesh": 13.1,
    "Jharkhand": 11.4, "Karnataka": 16.9, "Kerala": 7.3, "Madhya Pradesh": 16.6,
    "Maharashtra": 14.2, "Manipur": 20.4, "Meghalaya": 15.6, "Mizoram": 27.3,
    "Nagaland": 15.1, "Odisha": 11.3, "Punjab": 9.9, "Rajasthan": 12.3,
    "Uttarakhand": 14.7, "Uttar Pradesh": 8.9, "West Bengal": 9.0, "Delhi": 2.5,
    "Jammu & Kashmir": 25.4, "Ladakh": 51.8, "Lakshadweep": 1.7
}
state_keys = list(RADIAL_DISTANCES.keys())

# 3. GPS MATH
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    d_lat, d_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def find_nearest_district(target_district, lat, lon):
    min_dist, nearest = float('inf'), None
    for _, row in districts_geo.iterrows():
        if row['district'].lower() == target_district.lower(): continue
        dist = calculate_distance(lat, lon, row['Latitude'], row['Longitude'])
        if dist < 5.0: continue # Skip neighbors closer than 5km (likely same city)
        if dist < min_dist: 
            min_dist, nearest = dist, row['district']
    return nearest, min_dist

# 4. INTEL GATHERING
def safe_float(val, fallback):
    return fallback if pd.isna(val) or math.isnan(val) else float(val)

def get_district_intel(search_name):
    match, score = process.extractOne(search_name, all_districts)
    if score < 80: return None
    row_pop = df_pop[df_pop['district'] == match].iloc[0]
    row_water = df_water[df_water['district'] == match].mean(numeric_only=True)
    row_vax = df_vax[df_vax['district'] == match]
    
    raw_state = str(row_pop.get('state', 'Unknown'))
    
    # AUDIT FIX: MANUAL OVERRIDE FOR LEH/LADAKH
    # If the district is Leh, FORCE it to use Ladakh logic, ignoring the CSV state
    if "leh" in match.lower() or "ladakh" in match.lower():
        travel_dist = RADIAL_DISTANCES["Ladakh"]
    else:
        state_match, state_score = process.extractOne(raw_state, state_keys)
        travel_dist = RADIAL_DISTANCES.get(state_match, 12.0) if state_score > 80 else 12.0

    total_beds = max(1, int(safe_float(row_pop.get('estimated_total_beds'), 50)))

    return {
        "name": match,
        "state": raw_state,
        "travel_dist": travel_dist,
        "population": int(safe_float(row_pop.get('total_population'), 10000)),
        "beds": total_beds,
        "bod": safe_float(row_water.get('b.o.d. (mg/l)'), 1.0),
        "fecal": safe_float(row_water.get('fecal coliform (mpn/100ml)'), 10.0),
        "tds": safe_float(row_water.get('tds_level'), 150.0),
        "vax_rate": safe_float(row_vax.iloc[0].get('fully_vaccinated_%') if not row_vax.empty else None, 70.0)
    }

# 5. SIMULATION
def run_simulation(dist_name, month, current_rain, current_temp, prev_rain, prev_temp, lai, disease_name):
    intel = get_district_intel(dist_name)
    if not intel: return {"error": "Not Found"}
    if disease_name not in le.classes_: return {"error": "Disease Not Found"}

    # A. PREDICTION
    d_code = le.transform([disease_name])[0]
    features = ['month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp', 'bod', 'fecal_coliform', 'tds', 'vax_full', 'vax_measles', 'disease_code']
    input_vec = pd.DataFrame([[month, current_rain, current_temp, lai, prev_rain, prev_temp, intel['bod'], intel['fecal'], intel['tds'], intel['vax_rate'], intel['vax_rate'], d_code]], columns=features)
    
    raw_cases = int(model.predict(input_vec)[0])
    
    # BLACK SWAN LOGIC
    if current_rain > 500: raw_cases = int(raw_cases * 2.5)
    elif current_rain > 300: raw_cases = int(raw_cases * 1.5)
    
    predicted_cases = min(max(0, raw_cases), intel['population'])

    # B. XAI
    contribs = model.get_booster().predict(xgb.DMatrix(input_vec), pred_contribs=True)[0]
    feature_contributions = sorted(list(zip(features, contribs[:-1])), key=lambda x: abs(x[1]), reverse=True)
    readable_map = {'preci': 'Heavy Rainfall', 'prev_rain': 'Stagnant Water (Previous Week Rain)', 'fecal_coliform': 'Water Toxicity (Fecal)', 'vax_full': 'Low Immunization Shield', 'temp_celsius': 'Temperature Spikes', 'lai': 'Vegetation Density'}
    top_feature_raw = next(f for f, v in feature_contributions if f not in ['disease_code', 'month'])
    primary_driver = readable_map.get(top_feature_raw, top_feature_raw)

    # C. LOGISTICS
    penalty_factor = min(max(0, (intel['travel_dist'] - 10) * 0.02), 0.85)
    effective_beds = int(intel['beds'] * (1 - penalty_factor))
    hospital_demand = int(predicted_cases * 0.20)
    
    # D. INSIGHT
    people_per_bed = intel['population'] / intel['beds']
    insight = "✅ Infrastructure meets WHO standards."
    if people_per_bed > 1000:
        insight = f"⚠️ DECEPTIVE SURPLUS: 1 bed serves {int(people_per_bed)} people (WHO Standard: 1000). System overloaded."

    # E. OUTPUT
    print(f"\n{'='*70}")
    print(f"🏛️  GROUND ZERO: {intel['name'].upper()} ({disease_name})")
    print(f"{'='*70}")
    print(f"🔮 PREDICTION: {predicted_cases} Cases | Rate: {(predicted_cases/intel['population'])*10000:.2f}/10k")
    print(f"🧠 AI X-RAY: Driven by >> {primary_driver.upper()} <<")
    print(f"💡 INSIGHT: {insight}")
    
    print(f"\n⚙️  SYSTEM STRESS:")
    print(f"   - Travel Distance: {intel['travel_dist']} km (Logistics Penalty: -{int(penalty_factor*100)}%)")
    print(f"   - Effective Critical Beds: {effective_beds} (Raw: {intel['beds']})")
    print(f"   - Critical Demand: {hospital_demand}")

    status = "STABLE"
    if hospital_demand > effective_beds:
        overflow = hospital_demand - effective_beds
        print(f"🚨 STATUS: SYSTEM COLLAPSE (Deficit: {overflow})")
        status = "SYSTEM COLLAPSE"
        
        # SPILLOVER
        geo_row = districts_geo[districts_geo['district'] == intel['name']]
        lat = safe_float(geo_row.iloc[0]['Latitude'], 20.5)
        lon = safe_float(geo_row.iloc[0]['Longitude'], 78.9)
        nearest, distance = find_nearest_district(intel['name'], lat, lon)
        
        print(f"\n📡 SPILLOVER PROTOCOL:")
        print(f"   Redirecting {overflow} patients to {nearest} ({distance:.1f} km).")
        
        neighbor = get_district_intel(nearest)
        if neighbor:
            n_eff = int(neighbor['beds'] * (1 - min(max(0, (neighbor['travel_dist'] - 10) * 0.02), 0.85)))
            print(f"   >> {nearest} Effective Beds: {n_eff} | After Influx: {n_eff - overflow}")
            if (n_eff - overflow) < 0:
                print(f"   💥 DOMINO EFFECT: {nearest} HAS ALSO COLLAPSED.")
            else:
                print(f"   ⚠️ CONTAINED: {nearest} absorbed the shock.")
    else:
        print(f"✅ STATUS: STABLE")
    
    return {
        "district": intel['name'],
        "disease": disease_name,
        "prediction": predicted_cases,
        "status": status,
        "insight": insight,
        "spillover": {"target": nearest if hospital_demand > effective_beds else None}
    }

# SCENARIOS
print("\n[SYSTEM] Running Final Audited Scenarios...")
results = []
results.append(run_simulation("Dehradun", 8, 10, 28, 400, 30, 4.0, "Malaria"))
results.append(run_simulation("Leh", 7, 50, 18, 100, 16, 0.5, "Acute Diarrhoeal Disease"))
results.append(run_simulation("Aizawl", 6, 900, 32, 800, 30, 5.0, "Dengue"))

# Final JSON Export
with open('final_dashboard_payload.json', 'w') as f:
    json.dump(results, f, indent=4)