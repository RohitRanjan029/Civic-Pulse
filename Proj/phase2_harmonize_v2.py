import pandas as pd
import numpy as np
from fuzzywuzzy import process

# --- CONFIGURATION ---
HAZARD_FILE = '3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv'
VULNERABILITY_FILE = 'NFHS_5_India_Districts_Factsheet_Data.csv'
CAPACITY_FILE = 'RS_Session_257_AU_282_1.csv'
CLIMATE_FILE = 'climate_data.csv' 

print("--- PHASE 2.5: ADVANCED HARMONIZATION ---")

# 1. LOAD DATA
print("Loading datasets...")
df_hazard_raw = pd.read_csv(HAZARD_FILE)
df_vuln = pd.read_csv(VULNERABILITY_FILE)
df_cap = pd.read_csv(CAPACITY_FILE)

# Try loading climate data (Error handling if file format varies)
try:
    df_climate = pd.read_csv(CLIMATE_FILE)
    print(f"Climate Data Loaded: {df_climate.shape}")
except FileNotFoundError:
    print("CRITICAL WARNING: climate_data.csv not found. Creating dummy data for testing.")
    df_climate = pd.DataFrame({'District': df_vuln.iloc[:,0], 'temp': 30, 'humidity': 50, 'precip': 0})

# 2. RIGOROUS COLUMN CLEANING
# This fixes the KeyError you saw earlier by forcing special characters to underscores
def clean_cols(df):
    df.columns = [c.strip().lower().replace('\n', ' ').replace(' ', '_').replace('.', '').replace('-', '_').replace('/', '_') for c in df.columns]
    return df

df_vuln = clean_cols(df_vuln)
df_cap = clean_cols(df_cap)
df_hazard_raw = clean_cols(df_hazard_raw)
df_climate = clean_cols(df_climate)

# RENAME CRITICAL COLUMNS FOR STANDARDIZATION
# We force the join keys to simple names: 'state' and 'district'
df_vuln.rename(columns={'state_ut': 'state', 'district_names': 'district'}, inplace=True)
df_cap.rename(columns={'state_ut': 'state'}, inplace=True)
# Find the district column in climate data (it might be 'city', 'name', or 'district')
climate_cols = df_climate.columns.tolist()
if 'district' not in climate_cols:
    # Heuristic: The first column is usually the name
    df_climate.rename(columns={climate_cols[0]: 'district'}, inplace=True)

# 3. FIX STATE NAMES (The "Fuzzy" Logic)
valid_states = df_vuln['state'].unique()

def clean_state_name(name):
    if not isinstance(name, str): return "Unknown"
    name = name.replace('_', ' ')
    if "Jammu" in name: return "Jammu & Kashmir"
    if "Dadra" in name: return "Dadra and Nagar Haveli and Daman and Diu"
    
    # Fuzzy match
    match, score = process.extractOne(name, valid_states)
    if score > 85:
        return match
    return name

print("Normalizing State Names (this connects the dots)...")
df_hazard_raw['state'] = df_hazard_raw['state'].apply(clean_state_name)
df_cap['state'] = df_cap['state'].apply(clean_state_name)

# 4. AGGREGATE HAZARD DATA
# Sensor data is city-level. We aggregate to State level to fill gaps, 
# but if we had District mappings for stations, we would use that.
# For this Hackathon: State Average is the robust fallback.
df_hazard_agg = df_hazard_raw.pivot_table(
    index='state', 
    columns='pollutant_id', 
    values='pollutant_avg', 
    aggfunc='max' # Max is safer than Mean for risk detection (Worst case scenario)
).reset_index()

# Rename hazard columns so they don't clash
df_hazard_agg.columns = ['state' if x=='state' else f'haz_{x.lower()}' for x in df_hazard_agg.columns]

# 5. MERGE SEQUENCE (The "Intelligence" Layer)
# Start with Vulnerability (Base Layer) -> Join Infrastructure -> Join Hazard -> Join Climate
print("Merging Datasets...")

# Merge 1: Vuln + Capacity
df_master = pd.merge(df_vuln, df_cap, on='state', how='left')

# Merge 2: + Hazard (Pollution)
df_master = pd.merge(df_master, df_hazard_agg, on='state', how='left')

# Merge 3: + Climate (Weather)
# We assume Climate data is at District level. If not, we fall back.
df_master = pd.merge(df_master, df_climate, on='district', how='left')

# 6. FEATURE ENGINEERING FOR "UNSEEN PATTERNS"
# We create ratios. A high population with low hospitals is a "Hidden Risk"
# Convert columns to numeric, forcing errors to NaN then filling
cols_to_numeric = ['pop_below_15', 'phcs_total', 'chcs_total', 'haz_pm25', 'haz_no2', 'haz_so2']

# Rename specific NFHS columns to readable keys if they exist
# Note: I am using 'contains' logic to find the messy NFHS column names
def find_col(partial_name):
    matches = [c for c in df_master.columns if partial_name in c]
    return matches[0] if matches else None

col_child_pneumonia = find_col('ari') # Acute Respiratory Infection
col_stunting = find_col('stunted')
col_electricity = find_col('electricity')

print(f"Mapped Columns: ARI->{col_child_pneumonia}, Stunting->{col_stunting}")

# 7. FINAL CLEANUP
# Fill missing data with medians (Crucial for ML to not crash)
numeric_cols = df_master.select_dtypes(include=[np.number]).columns
df_master[numeric_cols] = df_master[numeric_cols].fillna(df_master[numeric_cols].median())

print(f"--- SUCCESS. MASTER TABLE SHAPE: {df_master.shape} ---")
df_master.to_csv('MASTER_Analytical_Base_Table.csv', index=False)
print("Saved to MASTER_Analytical_Base_Table.csv")