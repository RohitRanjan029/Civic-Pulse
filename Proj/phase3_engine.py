import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
INPUT_FILE = 'MASTER_Analytical_Base_Table.csv'

print("--- PHASE 3: ROBUST RISK ENGINE INITIALIZATION ---")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded Master Data: {df.shape}")

# 1. SMART COLUMN SELECTOR (The Fix)
# We search for columns by keywords instead of hardcoding exact names
def get_col_by_keyword(df, keywords):
    for col in df.columns:
        # Check if ALL keywords exist in the column name (case insensitive)
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

# Dynamically find the critical columns
col_district = get_col_by_keyword(df, ['district'])
col_pm25 = get_col_by_keyword(df, ['pm2.5']) or get_col_by_keyword(df, ['pm25'])
col_vuln_health = get_col_by_keyword(df, ['anaemia', 'child']) # Fallback to anaemia if ARI not found
col_vuln_ari = get_col_by_keyword(df, ['ari']) or get_col_by_keyword(df, ['pneumonia'])
col_cap_phc = get_col_by_keyword(df, ['phc', 'total'])
col_cap_chc = get_col_by_keyword(df, ['chc', 'total'])

# Climate columns
col_temp = get_col_by_keyword(df, ['temp'])
col_humid = get_col_by_keyword(df, ['humid'])

# Diagnostic Print
print("\n--- MAPPED FEATURES ---")
print(f"District Col: {col_district}")
print(f"Hazard (PM2.5): {col_pm25}")
print(f"Vuln (Health): {col_vuln_health}")
print(f"Vuln (ARI): {col_vuln_ari}")
print(f"Capacity (PHC): {col_cap_phc}")
print(f"Climate (Temp): {col_temp}")

# 2. FEATURE PREPARATION
# We use whatever we found. If something is missing, we drop it from the model list.
feature_candidates = [col_pm25, col_vuln_health, col_vuln_ari, col_cap_phc, col_cap_chc, col_temp, col_humid]
features = [f for f in feature_candidates if f is not None]

# Fill missing data
df_model = df[features].fillna(df[features].median())

# 3. THE AI MODEL: ISOLATION FOREST (Anomaly Detection)
# This finds the "Unseen Patterns" by isolating data points that are statistically unique
print(f"\nTraining AI on {len(features)} dimensions...")
iso_forest = IsolationForest(contamination=0.02, random_state=42) # Top 2% anomalies
df['anomaly_score'] = iso_forest.fit_predict(df_model)

# -1 is Anomaly (High Risk), 1 is Normal
df['risk_category'] = df['anomaly_score'].apply(lambda x: 'CRITICAL' if x == -1 else 'Standard')

# 4. DETERMINISTIC SCORING (The Logic Layer)
# We calculate a 'Strain Index' for sorting: (Hazard * Vulnerability) / Capacity
# Normalize first so math works (0 to 1 scale)
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_model), columns=df_model.columns)

# Safely calculate strain (avoiding divide by zero)
# Logic: (PM2.5 + Anaemia) / (PHCs + 1)
hazard_score = df_norm[col_pm25] if col_pm25 else 0.5
vuln_score = df_norm[col_vuln_health] if col_vuln_health else 0.5
cap_score = df_norm[col_cap_phc] if col_cap_phc else 0.5

df['calculated_strain'] = ((hazard_score + vuln_score) / (cap_score + 0.1)) * 10

# 5. GENERATE GOVERNMENT REPORT
def generate_report_entry(row):
    dist_name = row[col_district]
    
    # Get values safely
    val_pm25 = round(row[col_pm25], 1) if col_pm25 else "N/A"
    val_ari = round(row[col_vuln_ari], 1) if col_vuln_ari else "N/A"
    val_phc = int(row[col_cap_phc]) if col_cap_phc else "N/A"
    
    # Logic Checks
    is_anomaly = row['risk_category'] == 'CRITICAL'
    high_pollution = isinstance(val_pm25, float) and val_pm25 > 60
    low_capacity = isinstance(val_phc, int) and val_phc < 20
    
    # Construct Reasoning String
    reasons = []
    
    if is_anomaly:
        reasons.append("⚠️ UNSEEN PATTERN: ML Model detected distinct statistical anomaly (Outlier Event).")
    
    if high_pollution:
        reasons.append(f"⚠️ HAZARD ALERT: PM2.5 is Dangerous ({val_pm25}).")
        
    if high_pollution and low_capacity:
        reasons.append(f"❌ CASCADING FAILURE PREDICTED: High Pollution + Low Infrastructure ({val_phc} PHCs).")
    elif low_capacity:
        reasons.append(f"Capacity Warning: Only {val_phc} PHCs available.")
    else:
        reasons.append("✅ INFRASTRUCTURE OK: Sufficient PHC count.")

    # Action Logic
    action = "Monitor Surveillance Feeds."
    if "CASCADING" in str(reasons):
        action = "🚨 URGENT: Deploy Mobile Oxygen Units & Alert District Magistrate."
    elif is_anomaly:
        action = "Investigate: Unusual combination of risk factors detected."

    full_reason = " | ".join(reasons)

    print(f"District: {dist_name}")
    print(f"Hazard: PM2.5 {val_pm25} | Vulnerability: ARI {val_ari}% | Capacity: {val_phc} PHCs")
    print(f"Model Result: \"{full_reason} ACTION REQUIRED: {action}\"")
    print("-" * 80)

# 6. EXECUTION OUTPUT
print("\n" + "="*30 + " GOVERNMENT INTELLIGENCE REPORT " + "="*30 + "\n")

# Report 1: Top 3 ML Anomalies (The "Unseen")
anomalies = df[df['risk_category'] == 'CRITICAL'].sort_values('calculated_strain', ascending=False).head(3)
for _, row in anomalies.iterrows():
    generate_report_entry(row)

# Report 2: Specific Targets (Dehradun/Kanpur/Patna)
targets = ['Dehradun', 'Kanpur', 'Patna', 'Lucknow']
print("\n" + "="*30 + " TARGETED DISTRICT ANALYSIS " + "="*30 + "\n")
target_df = df[df[col_district].astype(str).str.contains('|'.join(targets), case=False, na=False)]

for _, row in target_df.iterrows():
    generate_report_entry(row)

# Export for Neo4j
output_cols = [col_district, 'calculated_strain', 'risk_category'] + features
df[output_cols].to_json('final_risk_predictions.json', orient='records')
print(f"\n[SYSTEM] Full predictions exported to 'final_risk_predictions.json' for React Dashboard.")