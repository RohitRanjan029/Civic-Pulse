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

print("--- PHASE 9d: THE IRONCLAD CASCADING ENGINE ---")

# 1. LOAD DATA & MODELS
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
state_keys = list(RADIAL_DISTANCES.keys()) # For fuzzy matching states

# 3. GPS SPILLOVER MATH
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
        if dist < min_dist: min_dist, nearest = dist, row['district']
    return nearest, min_dist

# 4. DATA EXTRACTION
def safe_float(val, fallback):
    return fallback if pd.isna(val) or math.isnan(val) else float(val)

def get_district_intel(search_name):
    match, score = process.extractOne(search_name, all_districts)
    if score < 80: return None
    row_pop = df_pop[df_pop['district'] == match].iloc[0]
    row_water = df_water[df_water['district'] == match].mean(numeric_only=True)
    row_vax = df_vax[df_vax['district'] == match]
    
    # BUG 1 FIX: Fuzzy Match the State to ensure we get the right logistics penalty
    raw_state = str(row_pop.get('state', 'Unknown'))
    state_match, state_score = process.extractOne(raw_state, state_keys)
    travel_dist = RADIAL_DISTANCES.get(state_match, 12.0) if state_score > 80 else 12.0

    return {
        "name": match,
        "state": raw_state,
        "travel_dist": travel_dist,
        "population": int(safe_float(row_pop.get('total_population'), 10000)),
        "beds": max(1, int(safe_float(row_pop.get('estimated_total_beds'), 10))),
        "bod": safe_float(row_water.get('b.o.d. (mg/l)'), 1.0),
        "fecal": safe_float(row_water.get('fecal coliform (mpn/100ml)'), 10.0),
        "tds": safe_float(row_water.get('tds_level'), 150.0),
        "vax_rate": safe_float(row_vax.iloc[0].get('fully_vaccinated_%') if not row_vax.empty else None, 70.0)
    }

# 5. THE MASTER SIMULATION
def run_simulation(dist_name, month, current_rain, current_temp, prev_rain, prev_temp, lai, disease_name):
    intel = get_district_intel(dist_name)
    if not intel: return
    if disease_name not in le.classes_: return

    # A. PREDICT VOLUME
    d_code = le.transform([disease_name])[0]
    features = ['month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp', 'bod', 'fecal_coliform', 'tds', 'vax_full', 'vax_measles', 'disease_code']
    input_vec = pd.DataFrame([[month, current_rain, current_temp, lai, prev_rain, prev_temp, intel['bod'], intel['fecal'], intel['tds'], intel['vax_rate'], intel['vax_rate'], d_code]], columns=features)
    
    # 1. Base Prediction from XGBoost
    raw_prediction = int(model.predict(input_vec)[0])
    
    # 2. BLACK SWAN LOGIC (The Exponential Multiplier)
    # If rain is apocalyptic (>500mm), disease spreads exponentially, not linearly.
    black_swan_multiplier = 1.0
    if current_rain > 500:
        black_swan_multiplier = 2.5 # Flood disaster multiplier
        print(f"⚠️ BLACK SWAN EVENT DETECTED: Rain > 500mm. Applying 2.5x Crisis Multiplier.")
    elif current_rain > 300:
        black_swan_multiplier = 1.5 # Severe flooding
    
    predicted_cases = int(raw_prediction * black_swan_multiplier)
    
    # 3. Zombie Cap (Safety Net)
    predicted_cases = min(max(0, predicted_cases), intel['population']) 

    # B. EXPLAINABLE AI
    contribs = model.get_booster().predict(xgb.DMatrix(input_vec), pred_contribs=True)[0]
    feature_contributions = sorted(list(zip(features, contribs[:-1])), key=lambda x: abs(x[1]), reverse=True)
    
    readable_map = {
        'preci': 'Current Heavy Rainfall', 'prev_rain': 'Stagnant Water (Previous Week Rain)',
        'fecal_coliform': 'High Fecal Water Toxicity', 'vax_full': 'Low Immunization Shield', 
        'temp_celsius': 'Current Heat Spike', 'prev_temp': 'Incubation Heat (Previous Week)',
        'bod': 'Biological Oxygen Demand (Water)', 'lai': 'Vegetation Density'
    }
    
    top_feature_raw = next(f for f, v in feature_contributions if f not in ['disease_code', 'month'])
    primary_driver = readable_map.get(top_feature_raw, top_feature_raw)

    # C. LOGISTICS PENALTY MATH
    penalty_factor = max(0, (intel['travel_dist'] - 10) * 0.02)
    
    # HARD CAP: Penalty cannot exceed 85% (Some people always make it to the hospital)
    penalty_factor = min(penalty_factor, 0.85) 
    
    effective_beds = int(intel['beds'] * (1 - penalty_factor))
    hospital_demand = int(predicted_cases * 0.20) # 20% need beds
    
    # D. DECEPTIVE SURPLUS LOGIC
    people_per_bed = intel['population'] / intel['beds']
    insight = "✅ Infrastructure meets WHO standards."
    if people_per_bed > 1500:
        insight = f"⚠️ DECEPTIVE SURPLUS: 1 bed serves {int(people_per_bed)} people (WHO Limit: 333). Chronic baseline overload."

    # E. CONSOLE OUTPUT
    print(f"\n{'='*70}")
    print(f"🏛️  GROUND ZERO: {intel['name'].upper()} ({disease_name} Outbreak)")
    print(f"{'='*70}")
    print(f"🔮 PREDICTION: {predicted_cases} Cases Predicted")
    print(f"🧠 AI X-RAY: Outbreak triggered primarily by >> {primary_driver.upper()} <<")
    print(f"💡 INSIGHT: {insight}")
    
    print(f"\n⚙️  SYSTEM STRESS CALCULUS:")
    print(f"   - Raw Bed Capacity: {intel['beds']}")
    if penalty_factor > 0:
        print(f"   - Logistics Penalty: -{int(penalty_factor*100)}% (Due to {intel['travel_dist']}km avg hospital distance)")
    print(f"   - Effective Beds: {effective_beds}")
    print(f"   - Critical Patient Demand: {hospital_demand}")

    if hospital_demand > effective_beds:
        overflow = hospital_demand - effective_beds
        print(f"🚨 STATUS: SYSTEM COLLAPSE (Deficit of {overflow} beds)")
        
        # GPS SPILLOVER
        geo_row = districts_geo[districts_geo['district'] == intel['name']]
        lat = safe_float(geo_row.iloc[0]['Latitude'] if not geo_row.empty else 20.5, 20.5)
        lon = safe_float(geo_row.iloc[0]['Longitude'] if not geo_row.empty else 78.9, 78.9)
        
        nearest_city, distance = find_nearest_district(intel['name'], lat, lon)
        print(f"\n📡 INITIATING SPATIAL SPILLOVER...")
        print(f"   Redirecting {overflow} critical patients to nearest hub: {nearest_city} ({distance:.1f} km away).")
        
        neighbor_intel = get_district_intel(nearest_city)
        if neighbor_intel:
            # Apply logistics penalty to the neighbor too!
            n_pen = min(max(0, (neighbor_intel['travel_dist'] - 10) * 0.02), 0.85)
            n_eff_beds = int(neighbor_intel['beds'] * (1 - n_pen))
            
            print(f"   >> {nearest_city} Effective Beds Before: {n_eff_beds} | After Influx: {n_eff_beds - overflow}")
            if (n_eff_beds - overflow) < 0:
                print(f"   💥 DOMINO EFFECT: {nearest_city} HAS ALSO COLLAPSED.")
            else:
                print(f"   ⚠️ CONTAINED: {nearest_city} absorbed the shock.")
    else:
        print(f"✅ STATUS: STABLE (Local infrastructure can absorb the surge)")


print("\n[SYSTEM] Executing Omni-Engine Simulations...")

# Scenario 1: Normal Dehradun
run_simulation("Dehradun", 8, current_rain=10, current_temp=28, prev_rain=400, prev_temp=30, lai=4.0, disease_name="Malaria")

# Scenario 2: Normal Leh (Checking if the 51km distance is fixed)
run_simulation("Leh", 7, current_rain=50, current_temp=18, prev_rain=100, prev_temp=16, lai=0.5, disease_name="Acute Diarrhoeal Disease")

# BUG 2 FIX: Scenario 3 - THE BLACK SWAN EVENT 
# A massive flood/cyclone hits Aizawl. We force the AI to process an apocalyptic amount of water.
run_simulation("Aizawl", 6, current_rain=900, current_temp=32, prev_rain=800, prev_temp=30, lai=5.0, disease_name="Dengue")