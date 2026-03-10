import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import math
import json
import warnings

warnings.filterwarnings("ignore")

print("--- PHASE 11: TWO-TIER HEALTHCARE DEFENSE SYSTEM ---")

# --- CONFIGURATION ---
MODEL_FILE = 'sovereign_brain_perfected.json'
ENCODER_FILE = 'disease_encoder_v2.npy'
POPULATION_FILE = 'System_Collapse_Baseline.csv'
WATER_FILE = 'Cleaned_Historical_Water_Quality.csv'
VAX_FILE = 'Cleaned_Immunization_History.csv'
EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv'
MANPOWER_FILE = 'Cleaned_Manpower_State_Stats.csv'
INFRA_FILE = 'MASTER_Analytical_Base_Table.csv' # Need this for exact PHC/CHC counts

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

# Cleanup
for df in [df_pop, df_water, df_vax, df_manpower, df_infra]:
    df.columns = [c.strip().lower() for c in df.columns]

# 2. INTELLIGENCE GATHERING (Updated for Tiered Data)
def get_tiered_intel(search_name):
    # Fuzzy Match District
    all_districts = df_pop['district'].unique()
    match, score = process.extractOne(search_name, all_districts)
    if score < 80: return None
    
    # Get Counts
    infra_row = df_infra[df_infra['district'] == match]
    if infra_row.empty:
        phc_count = 10
        chc_count = 2
    else:
        phc_count = int(infra_row.iloc[0].get('phcs_total', 10))
        chc_count = int(infra_row.iloc[0].get('chcs_total', 2))

    # Get Population & Manpower
    row_pop = df_pop[df_pop['district'] == match].iloc[0]
    pop = int(row_pop.get('total_population', 10000))
    state_name = row_pop.get('state', 'Unknown')
    
    # Get State Manpower Stats
    manpower_states = df_manpower['state'].unique()
    state_match, _ = process.extractOne(state_name, manpower_states)
    row_mp = df_manpower[df_manpower['state'] == state_match].iloc[0]
    
    # Estimate Specialists in District (Proportional to District Pop vs State Pop Proxy)
    # Using a heuristic: 1 district ~ 1/30th of state
    physicians = int(row_mp.get('physicians_in_position', 0) / 30)
    pediatricians = int(row_mp.get('paediatricians_in_position', 0) / 30)
    
    # Get Bio Data
    row_water = df_water[df_water['district'] == match].mean(numeric_only=True)
    row_vax = df_vax[df_vax['district'] == match]
    vax_rate = 70.0
    if not row_vax.empty: vax_rate = float(row_vax.iloc[0].get('fully_vaccinated_%', 70))

    return {
        "name": match,
        "state": state_name,
        "population": pop,
        "phc_count": max(1, phc_count),
        "chc_count": max(1, chc_count),
        "physicians": max(0, physicians),
        "pediatricians": max(0, pediatricians),
        "bod": float(row_water.get('b.o.d. (mg/l)', 1.0)),
        "fecal": float(row_water.get('fecal coliform (mpn/100ml)', 10.0)),
        "tds": float(row_water.get('tds_level', 150.0)),
        "vax_rate": vax_rate
    }

# 3. THE TWO-TIER SIMULATION
def run_tiered_simulation(dist_name, month, rain, temp, lai, disease_name):
    intel = get_tiered_intel(dist_name)
    if not intel: return {"error": "District Not Found"}
    if disease_name not in le.classes_: return {"error": "Disease Unknown"}

    # --- STEP 1: PREDICT TOTAL LOAD ---
    d_code = le.transform([disease_name])[0]
    input_vec = pd.DataFrame([[month, rain, temp, lai, rain, temp, intel['bod'], intel['fecal'], intel['tds'], intel['vax_rate'], intel['vax_rate'], d_code]], 
                             columns=['month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp', 'bod', 'fecal_coliform', 'tds', 'vax_full', 'vax_measles', 'disease_code'])
    
    raw_cases = int(model.predict(input_vec)[0])
    if rain > 500: raw_cases = int(raw_cases * 3.0) # Aggressive Black Swan
    elif rain > 300: raw_cases = int(raw_cases * 1.8)
    
    total_cases = min(max(0, raw_cases), intel['population'])
    
    # --- STEP 2: TRIAGE LOGIC ---
    # 80% Mild -> Go to PHC (Tier 1)
    # 20% Severe -> Go to CHC (Tier 2) immediately
    mild_cases = int(total_cases * 0.80)
    severe_cases = int(total_cases * 0.20)

    # --- STEP 3: TIER 1 ANALYSIS (PHC) ---
    # Capacity: 1 PHC can handle ~40 OPD patients/day effectively
    phc_capacity = intel['phc_count'] * 40
    phc_status = "✅ STABLE"
    phc_overflow = 0
    
    if mild_cases > phc_capacity:
        phc_status = "⚠️ OVERRUN"
        phc_overflow = mild_cases - phc_capacity # These people panic and run to the CHC
    
    # --- STEP 4: TIER 2 ANALYSIS (CHC) ---
    # Load = The Severe Cases + The Panic Overflow from PHCs
    chc_load = severe_cases + phc_overflow
    
    # Capacity: 1 CHC can handle ~100 patients/day
    chc_capacity = intel['chc_count'] * 100
    chc_status = "✅ STABLE"
    
    # --- STEP 5: MANPOWER CHECK (The Doctor Bottleneck) ---
    # Determine which specialist is needed
    specialist_name = "Physician"
    specialist_count = intel['physicians']
    if "Diarrhoeal" in disease_name:
        specialist_name = "Pediatrician"
        specialist_count = intel['pediatricians']
    
    # Manpower Ratio at CHC level
    # If 0 specialists, it's an immediate collapse for severe cases
    doc_status = "✅ AVAILABLE"
    if specialist_count == 0:
        doc_status = f"🚨 ZERO {specialist_name.upper()}S AVAILABLE"
        chc_status = "🚨 COLLAPSE (No Doctors)"
    elif (chc_load / specialist_count) > 100:
        doc_status = f"⚠️ OVERLOADED (1 Doc : {int(chc_load/specialist_count)} Patients)"
        if chc_load > chc_capacity: chc_status = "🚨 COLLAPSE"
    
    if chc_load > chc_capacity and chc_status != "🚨 COLLAPSE":
        chc_status = "⚠️ CRITICAL STRESS"

    # --- OUTPUT REPORT ---
    print(f"\n{'='*70}")
    print(f"🏥 TWO-TIER HEALTH DEFENSE: {intel['name'].upper()}")
    print(f"{'='*70}")
    print(f"1. CAUSE: Weather ({rain}mm) + {disease_name} Risk")
    print(f"2. EFFECT: {total_cases} Total Predicted Patients")
    print("-" * 70)
    
    print(f"🛡️  TIER 1 (VILLAGE LEVEL - PHC):")
    print(f"   - Facilities: {intel['phc_count']} PHCs")
    print(f"   - Patient Load: {mild_cases} (Mild Cases)")
    print(f"   - Capacity: {phc_capacity}")
    if phc_overflow > 0:
        print(f"   ❌ STATUS: {phc_status} -> {phc_overflow} patients spilling to Tier 2.")
    else:
        print(f"   ✅ STATUS: {phc_status} -> Contained at village level.")

    print("-" * 70)
    print(f"🛡️  TIER 2 (BLOCK LEVEL - CHC):")
    print(f"   - Facilities: {intel['chc_count']} CHCs")
    print(f"   - Patient Load: {chc_load} (Severe + Overflow)")
    print(f"   - Specialist Check: {specialist_count} {specialist_name}s")
    print(f"   - Manpower Status: {doc_status}")
    
    if "COLLAPSE" in chc_status:
        print(f"   🚨 FINAL VERDICT: {chc_status}")
        print(f"   📝 ORDER: BYPASS CHC. AIRLIFT PATIENTS TO DISTRICT HOSPITAL.")
    else:
        print(f"   ✅ FINAL VERDICT: {chc_status}")
        print(f"   📝 ORDER: Standard Protocol.")

# RUN SCENARIOS
print("\n[SYSTEM] Running Multi-Tier Simulation...")
run_tiered_simulation("Kanpur Nagar", 8, 300, 32, 2.5, "Dengue")
run_tiered_simulation("Leh", 7, 50, 18, 0.5, "Acute Diarrhoeal Disease")
# Massive disaster to force Tier 1 failure
run_tiered_simulation("Aizawl", 6, 900, 32, 5.0, "Dengue")