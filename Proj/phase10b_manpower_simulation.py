import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import math
import json
import warnings

warnings.filterwarnings("ignore")

print("--- PHASE 10b: THE ULTIMATE MANPOWER & RISK ENGINE ---")

# --- CONFIGURATION ---
MODEL_FILE = 'sovereign_brain_perfected.json'
ENCODER_FILE = 'disease_encoder_v2.npy'
POPULATION_FILE = 'System_Collapse_Baseline.csv'
WATER_FILE = 'Cleaned_Historical_Water_Quality.csv'
VAX_FILE = 'Cleaned_Immunization_History.csv'
EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv'
MANPOWER_FILE = 'Cleaned_Manpower_State_Stats.csv'

# 1. LOAD ALL DATASETS
print("[SYSTEM] Loading Neural Brain & National Health Grid...")
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
le = LabelEncoder()
le.classes_ = np.load(ENCODER_FILE, allow_pickle=True)

df_pop = pd.read_csv(POPULATION_FILE)
df_water = pd.read_csv(WATER_FILE)
df_vax = pd.read_csv(VAX_FILE)
df_geo = pd.read_csv(EPICLIM_FILE)
df_manpower = pd.read_csv(MANPOWER_FILE)

# Column Cleanup
for df in [df_pop, df_water, df_vax, df_manpower]:
    df.columns = [c.strip().lower() for c in df.columns]

# 2. PRE-CALCULATE STATE POPULATIONS
# We need this to calculate "Doctors per Capita" for the state
state_populations = df_pop.groupby('state')['total_population'].sum().to_dict()
manpower_states = df_manpower['state'].unique()

# 3. HELPER FUNCTIONS
def safe_float(val, fallback):
    return fallback if pd.isna(val) or math.isnan(val) else float(val)

def get_manpower_stats(state_name, district_pop):
    # Fuzzy match state name (e.g. "Andhra" -> "Andhra Pradesh")
    match, score = process.extractOne(state_name, manpower_states)
    if score < 80: return None
    
    row = df_manpower[df_manpower['state'] == match].iloc[0]
    
    # Get State Population (Sum of districts)
    # Fallback to 50 million if state not found in pop file
    state_pop = state_populations.get(state_name.lower(), 50000000) 
    
    # Calculate Ratios
    def est_district_staff(col_name):
        state_count = safe_float(row.get(col_name), 0)
        # Ratio: (State_Docs / State_Pop) * District_Pop
        return int((state_count / state_pop) * district_pop)

    return {
        "phc_docs": est_district_staff('phc_docs_in_position'),
        "physicians": est_district_staff('physicians_in_position'),
        "pediatricians": est_district_staff('paediatricians_in_position'),
        "total_docs": est_district_staff('total_doctors_available'),
        "shortfall_physician_pct": safe_float(row.get('physicians_shortfall_pct'), 0),
        "shortfall_pedia_pct": safe_float(row.get('paediatricians_shortfall_pct'), 0)
    }

def get_district_intel(search_name):
    # Find District
    all_districts = df_pop['district'].unique()
    match, score = process.extractOne(search_name, all_districts)
    if score < 80: return None
    
    row_pop = df_pop[df_pop['district'] == match].iloc[0]
    state_name = row_pop.get('state', 'Unknown')
    
    # Get Environmental Data
    row_water = df_water[df_water['district'] == match].mean(numeric_only=True)
    row_vax = df_vax[df_vax['district'] == match]
    vax_rate = safe_float(row_vax.iloc[0].get('fully_vaccinated_%') if not row_vax.empty else None, 70.0)
    
    pop = int(safe_float(row_pop.get('total_population'), 10000))
    
    # Get Manpower
    manpower = get_manpower_stats(state_name, pop)
    
    return {
        "name": match,
        "state": state_name,
        "population": pop,
        "beds": max(1, int(safe_float(row_pop.get('estimated_total_beds'), 10))),
        "bod": safe_float(row_water.get('b.o.d. (mg/l)'), 1.0),
        "fecal": safe_float(row_water.get('fecal coliform (mpn/100ml)'), 10.0),
        "tds": safe_float(row_water.get('tds_level'), 150.0),
        "vax_rate": vax_rate,
        "manpower": manpower
    }

# 4. THE SIMULATION
def run_simulation(dist_name, month, rain, temp, lai, disease_name):
    intel = get_district_intel(dist_name)
    if not intel: return
    if disease_name not in le.classes_: return

    # A. PREDICT CASE VOLUME
    d_code = le.transform([disease_name])[0]
    features = ['month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp', 'bod', 'fecal_coliform', 'tds', 'vax_full', 'vax_measles', 'disease_code']
    input_vec = pd.DataFrame([[month, rain, temp, lai, rain, temp, intel['bod'], intel['fecal'], intel['tds'], intel['vax_rate'], intel['vax_rate'], d_code]], columns=features)
    
    raw_cases = int(model.predict(input_vec)[0])
    
    # Black Swan Multiplier
    if rain > 500: raw_cases = int(raw_cases * 2.5)
    elif rain > 300: raw_cases = int(raw_cases * 1.5)
    predicted_cases = min(max(0, raw_cases), intel['population'])

    # B. MANPOWER REALITY CHECK (The "Sherlock" Logic)
    mp = intel['manpower']
    if not mp: 
        doc_insight = "⚠️ No Manpower Data Available."
    else:
        # 1. Determine relevant specialist
        required_specialist = "General Doctors"
        available_count = mp['total_docs'] # Default to all docs
        shortfall_pct = 0
        
        if "Diarrhoeal" in disease_name or "Child" in disease_name:
            required_specialist = "Paediatricians"
            available_count = mp['pediatricians']
            shortfall_pct = mp['shortfall_pedia_pct']
        elif "Dengue" in disease_name or "Malaria" in disease_name:
            required_specialist = "Physicians (MD)"
            available_count = mp['physicians'] + mp['phc_docs'] # PHC docs handle malaria
            shortfall_pct = mp['shortfall_physician_pct']

        # 2. Calculate "Deceptive Surplus" Ratio
        # WHO Goal: 1 Doc per 1000 people.
        # If available_count is 0, set to 1 to avoid div/0
        doc_count = max(1, available_count)
        people_per_doc = intel['population'] / doc_count
        
        doc_insight = f"✅ Workforce Stable (1 Doc : {int(people_per_doc)} people)"
        
        if people_per_doc > 5000:
            doc_insight = f"🚨 CRITICAL MANPOWER COLLAPSE: 1 {required_specialist} is forced to serve {int(people_per_doc)} people (Target: 1000). State Shortfall: {shortfall_pct}%."
        elif people_per_doc > 1500:
            doc_insight = f"⚠️ DECEPTIVE SURPLUS: Posts may be filled, but load is dangerous (1 Doc : {int(people_per_doc)} people). Patients will face extreme wait times."

    # C. BED CAPACITY CHECK
    hospital_demand = int(predicted_cases * 0.20)
    bed_deficit = hospital_demand - intel['beds']
    
    status = "STABLE"
    if bed_deficit > 0: status = "SYSTEM COLLAPSE"
    elif hospital_demand > (intel['beds'] * 0.8): status = "CRITICAL STRESS"

    # D. OUTPUT REPORT
    print(f"\n{'='*70}")
    print(f"🏛️  SOVEREIGN INTELLIGENCE: {intel['name'].upper()}, {intel['state'].upper()}")
    print(f"{'='*70}")
    print(f"🌊 SCENARIO: {disease_name} Outbreak | Weather: {rain}mm Rain")
    print(f"🔮 PREDICTION: {predicted_cases} Cases Predicted")
    
    print(f"\n🩺 MANPOWER REALITY CHECK (WHO Standard 1:1000):")
    print(f"   - Target Specialist: {required_specialist}")
    print(f"   - Estimated Available: {max(1, available_count)} doctors")
    print(f"   - {doc_insight}")
    
    print(f"\n🏥 INFRASTRUCTURE LOAD:")
    print(f"   - Critical Patients: {hospital_demand}")
    print(f"   - Total Beds: {intel['beds']}")
    
    if status == "SYSTEM COLLAPSE":
        print(f"   🚨 STATUS: COLLAPSE (Deficit: {bed_deficit} beds)")
        print(f"   📝 ACTION: Deploy Mobile Medical Units immediately.")
    else:
        print(f"   ✅ STATUS: STABLE")

# RUN SCENARIOS
print("\n[SYSTEM] Initiating Governance Audits...")

# Scenario 1: Kanpur - Dengue (Needs Physicians)
run_simulation("Kanpur Nagar", 8, 300, 32, 2.5, "Dengue")

# Scenario 2: Leh - Diarrhea (Needs Pediatricians)
run_simulation("Leh", 7, 50, 18, 0.5, "Acute Diarrhoeal Disease")

# Scenario 3: Aizawl - Massive Outbreak
run_simulation("Aizawl", 6, 900, 32, 5.0, "Dengue")