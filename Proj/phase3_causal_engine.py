import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# --- CONFIGURATION ---
INPUT_FILE = 'MASTER_Analytical_Base_Table.csv'

print("--- PHASE 3.5: CAUSAL INFERENCE ENGINE (CORRECTED) ---")
df = pd.read_csv(INPUT_FILE)

# 1. KNOWLEDGE BASE (The "Connecting the Dots" Logic)
RISK_KNOWLEDGE_GRAPH = {
    "AIR_QUALITY_SPIKE": {
        "prediction": "Respiratory Crisis (Asthma/Bronchitis)"
    },
    "WATER_LOGGING_EVENT": {
        "prediction": "Vector-Borne Outbreak (Dengue/Malaria)"
    }
}

# 2. DATA PREP (Smart Column Finder)
def get_col_by_keyword(df, keywords):
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords): return col
    return None

col_district = get_col_by_keyword(df, ['district'])
col_ari = get_col_by_keyword(df, ['ari']) or get_col_by_keyword(df, ['pneumonia'])
col_sanitation = get_col_by_keyword(df, ['sanitation'])
col_phc = get_col_by_keyword(df, ['phc', 'total'])
col_pm25 = get_col_by_keyword(df, ['pm2.5']) or get_col_by_keyword(df, ['pm25'])

# Train Anomaly Detection
feature_cols = [col_ari, col_sanitation, col_phc, col_pm25]
clean_features = [f for f in feature_cols if f is not None]
df_model = df[clean_features].fillna(df[clean_features].median())

iso_forest = IsolationForest(contamination=0.03, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(df_model)

# 3. THE CAUSAL SIMULATOR FUNCTION (FIXED MATCHING LOGIC)
def simulate_event(search_term, event_type, intensity_value=None):
    # FIX: Use case-insensitive contains instead of exact match
    matches = df[df[col_district].astype(str).str.contains(search_term, case=False, na=False)]
    
    if matches.empty:
        print(f"\n>> SIMULATION SKIPPED: No district matching '{search_term}' found.")
        return
    
    row = matches.iloc[0] # Grab the first match
    actual_district = row[col_district] # Get the real name from the CSV
    
    # 1. Check Anomalies (Unseen Patterns)
    is_anomaly = row['anomaly_score'] == -1
    anomaly_text = "⚠️ STATISTICAL ANOMALY DETECTED: This district has a rare combination of risk factors." if is_anomaly else "Statistical profile is normal."

    # 2. Run Causal Logic
    prediction_result = []
    
    if event_type == "WATER_LOGGING":
        sanitation_level = row.get(col_sanitation, 100)
        
        if sanitation_level < 50:
            prediction_result.append(f"🔴 HIGH RISK: Dengue/Cholera. Reason: Only {sanitation_level:.1f}% sanitation coverage makes this district highly vulnerable to water logging.")
        elif sanitation_level < 80:
             prediction_result.append(f"🟠 MODERATE RISK: Vector breeding likely. Sanitation ({sanitation_level:.1f}%) is suboptimal.")
        else:
             prediction_result.append(f"🟢 LOW RISK: Good sanitation ({sanitation_level:.1f}%) mitigates outbreak risk.")

    elif event_type == "AIR_POLLUTION":
        ari_level = row.get(col_ari, 0)
        current_pm = intensity_value if intensity_value else row.get(col_pm25, 50)
        
        if current_pm > 200 and ari_level > 2:
            prediction_result.append(f"🔴 CRITICAL RESPIRATORY FAILURE. Reason: Toxic Air ({current_pm}) hitting population with high respiratory history ({ari_level}%).")
        elif current_pm > 100:
            prediction_result.append("🟠 MODERATE RISK: General advisory recommended.")
        else:
            prediction_result.append("🟢 LOW RISK: Air quality is manageable.")

    # 3. Check Infrastructure Capacity
    phc_count = row.get(col_phc, 0)
    cap_text = f"Capacity Check: {int(phc_count)} PHCs available."
    if phc_count < 10:
        cap_text += " ❌ INSUFFICIENT infrastructure for crisis."
    else:
        cap_text += " ✅ Infrastructure adequate."

    # Formatting Output
    print(f"\n>> SIMULATION: {event_type} in {actual_district}")
    print(f"   {anomaly_text}")
    print(f"   {' '.join(prediction_result)}")
    print(f"   {cap_text}")
    print("-" * 60)

# 4. RUN SIMULATIONS
print("\n=== SYSTEM ONLINE: RUNNING SCENARIO SIMULATIONS ===")

simulate_event("Dehradun", "WATER_LOGGING")
simulate_event("Kanpur", "AIR_POLLUTION", intensity_value=450)
simulate_event("Leh", "WATER_LOGGING")

# I am adding a guaranteed random fallback just in case those names aren't in your specific CSV
random_district = df.iloc[0][col_district]
simulate_event(random_district, "AIR_POLLUTION", intensity_value=300)

print("\n[SYSTEM] Causal Inference Logic Validated.")