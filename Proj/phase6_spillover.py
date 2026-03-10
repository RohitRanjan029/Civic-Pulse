import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process

# --- CONFIGURATION ---
MODEL_FILE = 'epilim_brain.json'
ENCODER_FILE = 'disease_encoder.npy'
INFRA_FILE = 'MASTER_Analytical_Base_Table.csv'

print("--- PHASE 6.5: SPATIAL SPILLOVER ENGINE (THE DOMINO EFFECT) ---")

model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
encoder_classes = np.load(ENCODER_FILE, allow_pickle=True)
le = LabelEncoder()
le.classes_ = encoder_classes
df_infra = pd.read_csv(INFRA_FILE)
infra_districts = df_infra['district'].unique()

def get_district_capacity(search_name):
    match, score = process.extractOne(search_name, infra_districts)
    if score < 80: return None
    row = df_infra[df_infra['district'] == match].iloc[0]
    
    def get_val(keywords):
        for col in df_infra.columns:
            if all(k in col for k in keywords): return row[col]
        return 0

    # UPGRADED: Using severe bed capacity (Beds for critical IV patients, not just walk-ins)
    return {
        "name": match,
        "critical_beds": int((get_val(['phc', 'total']) * 2) + (get_val(['chc', 'total']) * 15)),
        "sanitation": get_val(['sanitation'])
    }

# WE DEFINE A GEOGRAPHICAL NEIGHBOR MAP (For the domino effect)
NEIGHBORS = {
    "Kanpur Nagar": "Lucknow",
    "Leh(Ladakh)": "Kargil",
    "Dehradun": "Haridwar"
}

def run_spillover_simulation(dist_name, lat, lon, month, rain, temp, lai, disease_name):
    try:
        d_code = le.transform([disease_name])[0]
    except:
        return
        
    # 1. Predict Demand
    input_vec = pd.DataFrame([[month, lat, lon, rain, temp, lai, rain, temp, d_code]], 
                             columns=['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code'])
    pred_cases = max(0, int(model.predict(input_vec)[0]))

    # 2. Get Supply
    infra = get_district_capacity(dist_name)
    if not infra: return

    capacity = infra['critical_beds'] if infra['critical_beds'] > 0 else 50
    
    # 3. Calculate Collision
    print(f"\n🔴 GROUND ZERO: {dist_name} ({disease_name} Outbreak)")
    print(f"   Cases: {pred_cases} | Available Critical Beds: {capacity}")
    
    if pred_cases > capacity:
        overflow = pred_cases - capacity
        print(f"   🚨 SYSTEM COLLAPSE: Exceeded capacity by {overflow} patients!")
        
        # 4. TRIGGER THE DOMINO EFFECT (Spillover to neighbor)
        neighbor = NEIGHBORS.get(dist_name)
        if neighbor:
            neighbor_infra = get_district_capacity(neighbor)
            n_cap = neighbor_infra['critical_beds'] if neighbor_infra else 100
            print(f"   🌊 SPILLOVER EVENT: {overflow} patients fleeing to neighboring {neighbor}...")
            print(f"   >> {neighbor} Beds Before: {n_cap} | After Influx: {n_cap - overflow}")
            if (n_cap - overflow) < 0:
                print(f"   💥 CASCADING FAILURE: {neighbor} hospital system has now also collapsed!")
            else:
                print(f"   ⚠️ {neighbor} is absorbing the shock, but is under critical stress.")
        else:
             print("   ⚠️ No neighboring district data mapped. Total isolation failure.")
    else:
        print("   ✅ System Stable. Local infrastructure can handle the load.")

# RUN SIMULATION
run_spillover_simulation("Kanpur Nagar", 26.4, 80.3, 8, 350.0, 32.0, 2.5, "Dengue")
run_spillover_simulation("Leh", 34.1, 77.5, 7, 50.0, 18.0, 0.5, "Acute Diarrhoeal Disease")