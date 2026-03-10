import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import math

# --- CONFIGURATION ---
MODEL_FILE = 'epilim_brain.json'
ENCODER_FILE = 'disease_encoder.npy'
INFRA_FILE = 'MASTER_Analytical_Base_Table.csv'
EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv' # We need this to get everyone's Lat/Lon

print("--- PHASE 7: GEOSPATIAL CASCADING ENGINE (DYNAMIC SPILLOVER) ---")

# 1. LOAD MODELS & DATA
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
le = LabelEncoder()
le.classes_ = np.load(ENCODER_FILE, allow_pickle=True)

df_infra = pd.read_csv(INFRA_FILE)
df_epiclim = pd.read_csv(EPICLIM_FILE)

# Extract unique districts and their Lat/Lon from EpiClim
districts_geo = df_epiclim.groupby('district').agg({'Latitude': 'first', 'Longitude': 'first'}).reset_index()

# 2. THE HAVERSINE FORMULA (Calculates exact distance between two GPS coordinates)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# 3. DYNAMIC NEIGHBOR FINDER
def find_nearest_district(target_district, lat, lon):
    min_dist = float('inf')
    nearest = None
    
    for _, row in districts_geo.iterrows():
        name = row['district']
        if name.lower() == target_district.lower(): continue # Skip self
        
        dist = calculate_distance(lat, lon, row['Latitude'], row['Longitude'])
        if dist < min_dist:
            min_dist = dist
            nearest = name
            
    return nearest, min_dist

# 4. INFRASTRUCTURE LOOKUP (Fixed to avoid state-bleeding)
infra_districts = df_infra['district'].unique()
def get_district_capacity(search_name):
    match, score = process.extractOne(search_name, infra_districts)
    if score < 80: return None
    row = df_infra[df_infra['district'] == match].iloc[0]
    
    # We cap the beds at a realistic district maximum so states don't bleed into cities
    phcs = min(row.get('phcs_total', 20), 100) # Fallback heuristic
    chcs = min(row.get('chcs_total', 5), 30)
    
    return {
        "name": match,
        "critical_beds": int((phcs * 2) + (chcs * 15)),
        "sanitation": row.get('vuln_sanitation', 50)
    }

# 5. THE GEOSPATIAL SIMULATOR
def run_dynamic_simulation(dist_name, lat, lon, month, rain, temp, lai, disease_name):
    try: d_code = le.transform([disease_name])[0]
    except: return
        
    # Predict Demand
    input_vec = pd.DataFrame([[month, lat, lon, rain, temp, lai, rain, temp, d_code]], 
                             columns=['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code'])
    pred_cases = max(0, int(model.predict(input_vec)[0]))

    # Get Supply
    infra = get_district_capacity(dist_name)
    if not infra: capacity = 50 
    else: capacity = infra['critical_beds']
    
    print(f"\n🔴 GROUND ZERO: {dist_name} ({disease_name} Outbreak)")
    print(f"   Predicted Critical Cases: {pred_cases} | Available Critical Beds: {capacity}")
    
    if pred_cases > capacity:
        overflow = pred_cases - capacity
        print(f"   🚨 SYSTEM COLLAPSE: Exceeded capacity by {overflow} patients!")
        
        # FIND NEAREST DISTRICT DYNAMICALLY
        nearest_city, distance = find_nearest_district(dist_name, lat, lon)
        print(f"   📡 SCANNING FOR NEAREST HOSPITAL INFRASTRUCTURE...")
        print(f"   📍 Found {nearest_city} at a distance of {distance:.1f} km.")
        
        neighbor_infra = get_district_capacity(nearest_city)
        n_cap = neighbor_infra['critical_beds'] if neighbor_infra else 100
        
        print(f"   🌊 SPILLOVER EVENT: {overflow} patients fleeing to {nearest_city}.")
        print(f"   >> {nearest_city} Beds Before: {n_cap} | After Influx: {n_cap - overflow}")
        
        if (n_cap - overflow) < 0:
            print(f"   💥 CASCADING FAILURE: The healthcare system in {nearest_city} has also collapsed!")
        else:
            print(f"   ⚠️ {nearest_city} absorbed the shock, but is under critical stress.")
    else:
        print("   ✅ System Stable. Local infrastructure can handle the load.")

# RUN
run_dynamic_simulation("Leh", 34.1, 77.5, 7, 50.0, 18.0, 0.5, "Acute Diarrhoeal Disease")
# Testing a random place in the Northeast
run_dynamic_simulation("Aizawl", 23.7, 92.7, 6, 200.0, 28.0, 3.0, "Malaria")