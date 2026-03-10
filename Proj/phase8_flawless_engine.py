import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import math
import json

# --- CONFIGURATION ---
MODEL_FILE = 'epilim_brain.json'
ENCODER_FILE = 'disease_encoder.npy'
INFRA_FILE = 'MASTER_Analytical_Base_Table.csv'
EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv' 

print("--- PHASE 8: THE FLAWLESS CASCADING ENGINE ---")
print("[SYSTEM] Initializing XGBoost Brain, AQI Respiratory Engine, and Spatial Spillover...\n")

# 1. LOAD MODELS & DATA
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
le = LabelEncoder()
le.classes_ = np.load(ENCODER_FILE, allow_pickle=True)

df_infra = pd.read_csv(INFRA_FILE)
df_epiclim = pd.read_csv(EPICLIM_FILE)
districts_geo = df_epiclim.groupby('district').agg({'Latitude': 'first', 'Longitude': 'first'}).reset_index()

# 2. THE KNOWLEDGE BASE: RADIAL DISTANCES (Extracted from Table 59 of your PDF)
# Average distance in KM a patient must travel to reach a Community Health Centre (CHC)
RADIAL_DISTANCES = {
    "Andhra Pradesh": 17.2, "Arunachal Pradesh": 21.6, "Assam": 11.0, "Bihar": 9.9,
    "Chhattisgarh": 15.9, "Gujarat": 13.0, "Haryana": 9.9, "Himachal Pradesh": 13.1,
    "Jharkhand": 11.4, "Karnataka": 16.9, "Kerala": 7.3, "Madhya Pradesh": 16.6,
    "Maharashtra": 14.2, "Manipur": 20.4, "Meghalaya": 15.6, "Mizoram": 27.3,
    "Nagaland": 15.1, "Odisha": 11.3, "Punjab": 9.9, "Rajasthan": 12.3,
    "Sikkim": 33.6, "Tamil Nadu": 9.8, "Telangana": 19.1, "Tripura": 12.6,
    "Uttarakhand": 14.7, "Uttar Pradesh": 8.9, "West Bengal": 9.0, "Delhi": 2.5,
    "Jammu & Kashmir": 25.4, "Ladakh": 51.8, "Lakshadweep": 1.7
}

# 3. HAVERSINE FORMULA (For Dynamic Spillover)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def find_nearest_district(target_district, lat, lon):
    min_dist = float('inf')
    nearest = None
    for _, row in districts_geo.iterrows():
        if row['district'].lower() == target_district.lower(): continue
        dist = calculate_distance(lat, lon, row['Latitude'], row['Longitude'])
        if dist < min_dist:
            min_dist = dist
            nearest = row['district']
    return nearest, min_dist

# 4. DISTRICT PROFILE EXTRACTOR
infra_districts = df_infra['district'].unique()
def get_district_profile(search_name):
    match, score = process.extractOne(search_name, infra_districts)
    if score < 80: return None
    row = df_infra[df_infra['district'] == match].iloc[0]
    
    def get_val(keywords):
        for col in df_infra.columns:
            if all(k in col for k in keywords): return row[col]
        return 0

    state = row.get('state', 'Unknown')
    travel_dist = RADIAL_DISTANCES.get(state, 12.0) # Default 12km if state not found
    
    return {
        "name": match,
        "state": state,
        "critical_beds": int((min(get_val(['phc', 'total']), 50) * 2) + (min(get_val(['chc', 'total']), 15) * 15)),
        "sanitation": get_val(['sanitation']),
        "ari_history": get_val(['ari']) or get_val(['pneumonia']),
        "travel_distance_km": travel_dist
    }

# 5. THE FLAWLESS SIMULATION ENGINE
def run_flawless_simulation(dist_name, lat, lon, month, rain, temp, lai, pm25_level, infectious_disease):
    profile = get_district_profile(dist_name)
    if not profile: return
    
    print(f"\n{'='*70}")
    print(f"🌍 DISTRICT OVERVIEW: {profile['name']}, {profile['state']}")
    print(f"🏥 CAPACITY: {profile['critical_beds']} Critical Beds | 🚑 AVG TRAVEL: {profile['travel_distance_km']} km")
    print(f"{'='*70}")

    # --- MODULE 1: INFECTION PREDICTION (XGBOOST) ---
    try: d_code = le.transform([infectious_disease])[0]
    except: d_code = 0
    input_vec = pd.DataFrame([[month, lat, lon, rain, temp, lai, rain, temp, d_code]], 
                             columns=['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code'])
    infectious_cases = max(0, int(model.predict(input_vec)[0]))

    # --- MODULE 2: RESPIRATORY PREDICTION (DETERMINISTIC AQI LOGIC) ---
    # Formula: If Air is toxic, multiply it by the population's historical vulnerability (ARI)
    respiratory_cases = 0
    if pm25_level > 100:
        ari_multiplier = max(profile['ari_history'], 1.0)
        respiratory_cases = int((pm25_level - 100) * ari_multiplier * 1.5)

    total_demand = infectious_cases + respiratory_cases

    print(f"🦠 THREAT 1 (Climate): {infectious_cases} predicted cases of {infectious_disease} (Rain: {rain}mm)")
    if pm25_level > 100:
        print(f"💨 THREAT 2 (Toxicity): {respiratory_cases} acute respiratory failures (PM2.5: {pm25_level}, Historical ARI: {profile['ari_history']}%)")
    print(f"📈 TOTAL PROJECTED PATIENT SURGE: {total_demand} patients.")

    # --- MODULE 3: THE LOGISTICS PENALTY ---
    # If a hospital is 50km away (like Ladakh), beds are useless because people die in transit.
    # We penalize effective capacity by 2% for every kilometer over 10km.
    penalty_factor = max(0, (profile['travel_distance_km'] - 10) * 0.02)
    effective_beds = int(profile['critical_beds'] * (1 - penalty_factor))
    
    print(f"\n⚙️  SYSTEM CALCULUS:")
    print(f"   - Raw Bed Capacity: {profile['critical_beds']}")
    if penalty_factor > 0:
        print(f"   - Logistics Penalty applied: -{int(penalty_factor*100)}% (Due to {profile['travel_distance_km']}km travel distance)")
    print(f"   - TRUE EFFECTIVE BEDS: {effective_beds}")

    # --- MODULE 4: COLLISION & SPILLOVER ---
    if total_demand > effective_beds:
        overflow = total_demand - effective_beds
        print(f"\n🚨 [CRITICAL ALERT]: LOCAL HEALTHCARE SYSTEM COLLAPSE.")
        print(f"   Deficit of {overflow} beds.")
        
        nearest_city, distance = find_nearest_district(dist_name, lat, lon)
        print(f"\n📡 INITIATING SPATIAL SPILLOVER PROTOCOL...")
        print(f"   Redirecting {overflow} critical patients to nearest hub: {nearest_city} ({distance:.1f} km away).")
        
        neighbor_profile = get_district_profile(nearest_city)
        n_beds = neighbor_profile['critical_beds'] if neighbor_profile else 100
        
        print(f"   >> {nearest_city} Capacity Before: {n_beds} | After Influx: {n_beds - overflow}")
        if (n_beds - overflow) < 0:
            print(f"   💥 [DOMINO EFFECT]: {nearest_city} HAS ALSO COLLAPSED.")
        else:
            print(f"   ⚠️ [CONTAINED]: {nearest_city} absorbed the overflow. Regional network heavily stressed.")
    else:
        print(f"\n✅ [STATUS NORMAL]: Local infrastructure can absorb the projected surge.")

# 6. RUN THE ULTIMATE SCENARIOS

# Scenario 1: Kanpur - High Pollution + Moderate Rain (Testing Respiratory Engine)
run_flawless_simulation("Kanpur Nagar", 26.4, 80.3, 8, 150.0, 32.0, 2.5, pm25_level=450, infectious_disease="Dengue")

# Scenario 2: Ladakh - Extreme Distance Logistics Penalty Test
run_flawless_simulation("Leh", 34.1, 77.5, 7, 50.0, 18.0, 0.5, pm25_level=60, infectious_disease="Acute Diarrhoeal Disease")

# Scenario 3: Dehradun - High Rain, Moderate Pollution, Checking Spillover
run_flawless_simulation("Dehradun", 30.3, 78.0, 8, 300.0, 26.0, 4.0, pm25_level=120, infectious_disease="Malaria")