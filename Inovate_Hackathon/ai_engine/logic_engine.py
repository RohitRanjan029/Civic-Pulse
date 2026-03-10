import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import math
import json
import warnings

warnings.filterwarnings("ignore")

print("--- FINAL OMEGA ENGINE: MULTI-TIER + MANPOWER + GEOSPATIAL ---")

# --- CONFIGURATION ---
MODEL_FILE = 'models/sovereign_brain_perfected.json'
ENCODER_FILE = 'models/disease_encoder_v2.npy'
POPULATION_FILE = 'data/System_Collapse_Baseline.csv'
WATER_FILE = 'data/Cleaned_Historical_Water_Quality.csv'
VAX_FILE = 'data/Cleaned_Immunization_History.csv'
EPICLIM_FILE = 'data/Cleaned_EpiClim_Data.csv'
MANPOWER_FILE = 'data/Cleaned_Manpower_State_Stats.csv'
INFRA_FILE = 'data/MASTER_Analytical_Base_Table.csv'

# 1. LOAD ARSENAL
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
le = LabelEncoder()
le.classes_ = np.load(ENCODER_FILE, allow_pickle=True)

df_pop = pd.read_csv(POPULATION_FILE)
df_water = pd.read_csv(WATER_FILE)
df_vax = pd.read_csv(VAX_FILE)
df_manpower = pd.read_csv(MANPOWER_FILE)
df_infra = pd.read_csv(INFRA_FILE)
df_geo = pd.read_csv(EPICLIM_FILE)

# Cleanup
for df in [df_pop, df_water, df_vax, df_manpower, df_infra, df_geo]:
    df.columns = [c.strip().lower() for c in df.columns]

# Geo Data
districts_geo = df_geo.groupby('district').agg({'latitude': 'first', 'longitude': 'first'}).reset_index()

# 2. GPS UTILITIES
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    d_lat, d_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def find_nearest_district(target_district, lat, lon):
    min_dist, nearest = float('inf'), None
    for _, row in districts_geo.iterrows():
        if row['district'].lower() == target_district.lower(): continue
        dist = calculate_distance(lat, lon, row['latitude'], row['longitude'])
        if dist < 5.0: continue # Skip self/twins
        if dist < min_dist: min_dist, nearest = dist, row['district']
    return nearest, min_dist

# 3. INTELLIGENCE GATHERING
def get_full_intel(search_name):
    # District Match
    all_districts = df_pop['district'].unique()
    match, score = process.extractOne(search_name, all_districts)
    if score < 80: return None
    
    # Infrastructure Counts
    infra_row = df_infra[df_infra['district'] == match]
    if infra_row.empty: phc_count, chc_count = 10, 2
    else:
        phc_count = int(infra_row.iloc[0].get('phcs_total', 10))
        chc_count = int(infra_row.iloc[0].get('chcs_total', 2))

    # Population & State
    row_pop = df_pop[df_pop['district'] == match].iloc[0]
    pop = int(row_pop.get('total_population', 10000))
    state_name = row_pop.get('state', 'Unknown')
    
    # Manpower
    manpower_states = df_manpower['state'].unique()
    state_match, _ = process.extractOne(state_name, manpower_states)
    row_mp = df_manpower[df_manpower['state'] == state_match].iloc[0]
    
    # Heuristic: 1 District = 1/30th of State Manpower
    physicians = int(row_mp.get('physicians_in_position', 0) / 30)
    pediatricians = int(row_mp.get('paediatricians_in_position', 0) / 30)
    
    # Bio Data
    row_water = df_water[df_water['district'] == match].mean(numeric_only=True)
    row_vax = df_vax[df_vax['district'] == match]
    vax_rate = float(row_vax.iloc[0].get('fully_vaccinated_%', 70)) if not row_vax.empty else 70.0

    return {
        "name": match, "state": state_name, "population": pop,
        "phc_count": max(1, phc_count), "chc_count": max(1, chc_count),
        "physicians": max(0, physicians), "pediatricians": max(0, pediatricians),
        "bod": float(row_water.get('b.o.d. (mg/l)', 1.0)),
        "fecal": float(row_water.get('fecal coliform (mpn/100ml)', 10.0)),
        "tds": float(row_water.get('tds_level', 150.0)),
        "vax_rate": vax_rate
    }

# 4. THE OMEGA SIMULATION
def run_omega_simulation(dist_name, month, rain, temp, lai, disease_name):
    intel = get_full_intel(dist_name)
    if not intel: return
    if disease_name not in le.classes_: return

    # --- STEP A: PREDICT ---
    d_code = le.transform([disease_name])[0]
    input_vec = pd.DataFrame([[month, rain, temp, lai, rain, temp, intel['bod'], intel['fecal'], intel['tds'], intel['vax_rate'], intel['vax_rate'], d_code]], 
                             columns=['month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp', 'bod', 'fecal_coliform', 'tds', 'vax_full', 'vax_measles', 'disease_code'])
    
    raw_cases = int(model.predict(input_vec)[0])
    if rain > 500: raw_cases = int(raw_cases * 3.0) # Black Swan
    
    total_cases = min(max(0, raw_cases), intel['population'])
    
    # --- STEP B: TIER 1 (PHC) ---
    mild_cases = int(total_cases * 0.80)
    phc_capacity = intel['phc_count'] * 40
    phc_status = "✅ STABLE"
    phc_overflow = 0
    
    if mild_cases > phc_capacity:
        phc_status = "⚠️ OVERRUN"
        phc_overflow = mild_cases - phc_capacity
    
    # --- STEP C: TIER 2 (CHC + MANPOWER) ---
    severe_cases = int(total_cases * 0.20)
    chc_load = severe_cases + phc_overflow
    chc_capacity = intel['chc_count'] * 100 # Bed Capacity
    
    # Doctor Check
    specialist = "Pediatrician" if "Diarrhoeal" in disease_name else "Physician"
    doc_count = intel['pediatricians'] if "Diarrhoeal" in disease_name else intel['physicians']
    
    chc_status = "✅ STABLE"
    reason = "Capacity Sufficient"
    
    if doc_count == 0:
        chc_status = "🚨 COLLAPSE"
        reason = f"Zero {specialist}s Available"
    elif chc_load > chc_capacity:
        chc_status = "🚨 COLLAPSE"
        reason = f"Bed Deficit ({chc_load - chc_capacity})"
    elif (chc_load / max(1, doc_count)) > 100:
        chc_status = "⚠️ CRITICAL STRESS"
        reason = "Doctor Overload"

    # --- STEP D: TIER 3 (GPS SPILLOVER) ---
    spillover_msg = "None"
    neighbor_data = None
    
    if "COLLAPSE" in chc_status:
        # Find Neighbor
        geo_row = districts_geo[districts_geo['district'] == intel['name']]
        lat = geo_row.iloc[0]['latitude'] if not geo_row.empty else 20.5
        lon = geo_row.iloc[0]['longitude'] if not geo_row.empty else 78.9
        
        nearest, dist = find_nearest_district(intel['name'], lat, lon)
        spillover_msg = f"Redirecting to {nearest} ({dist:.1f} km)"
        
        # Check Neighbor Capacity (Simplified)
        n_intel = get_full_intel(nearest)
        if n_intel:
            neighbor_data = {
                "name": nearest,
                "distance": f"{dist:.1f} km",
                "status": "ABSORBING" if n_intel['chc_count'] * 100 > chc_load else "CASCADING FAILURE"
            }

    # --- OUTPUT ---
    print(f"\n{'='*70}")
    print(f"🏥 OMEGA DEFENSE SYSTEM: {intel['name'].upper()}")
    print(f"{'='*70}")
    print(f"1. THREAT: {total_cases} Predicted Cases ({disease_name})")
    print(f"2. TIER 1 (PHC): {intel['phc_count']} units. Status: {phc_status}")
    if phc_overflow > 0: print(f"   >> {phc_overflow} patients spilling to Tier 2")
    
    print(f"3. TIER 2 (CHC): {intel['chc_count']} units. Status: {chc_status}")
    print(f"   >> Cause: {reason}")
    print(f"   >> Manpower: {doc_count} {specialist}s available.")
    
    if neighbor_data:
        print(f"4. TIER 3 (SPILLOVER): {spillover_msg}")
        print(f"   >> Target: {neighbor_data['name']} -> {neighbor_data['status']}")
    
    return {
        "district": intel['name'],
        "disease": disease_name,
        "prediction": total_cases,
        "phc_status": phc_status,
        "chc_status": chc_status,
        "spillover": neighbor_data,
        "action": "DEPLOY CENTRAL TEAM" if "COLLAPSE" in chc_status else "MONITOR"
    }

