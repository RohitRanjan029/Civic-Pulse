import pandas as pd
import numpy as np
from fuzzywuzzy import process

# --- CONFIGURATION ---
EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv' # Your main outbreak file (check filename!)
WATER_FILE = 'Cleaned_Historical_Water_Quality.csv'
VACCINE_FILE = 'Cleaned_Immunization_History.csv'

print("--- PHASE 9a: FUSING MULTI-DIMENSIONAL DATASETS ---")

# 1. LOAD DATA
print("Loading datasets...")
df_main = pd.read_csv(EPICLIM_FILE)
df_water = pd.read_csv(WATER_FILE)
df_vax = pd.read_csv(VACCINE_FILE)

# 2. STANDARDIZE COLUMN NAMES
# Ensure we can merge on 'district' and 'year'
df_main.columns = [c.strip().lower() for c in df_main.columns]
df_water.columns = [c.strip().lower() for c in df_water.columns]
df_vax.columns = [c.strip().lower() for c in df_vax.columns]

# Rename specific columns to be safe
# Map 'Cleaned_Historical_Water_Quality' columns to simple names
df_water = df_water.rename(columns={
    'b.o.d. (mg/l)': 'bod', 
    'fecal coliform (mpn/100ml)': 'fecal_coliform',
    'tds_level': 'tds'
})

# Map 'Cleaned_Immunization_History'
df_vax = df_vax.rename(columns={
    'fully_vaccinated_%': 'vax_full',
    'measles_coverage_%': 'vax_measles'
})

# 3. FUZZY MATCH DISTRICT NAMES (CRITICAL)
# Water/Vaccine data might spell "Ahmedabad" different from EpiClim
main_districts = df_main['district'].unique()

def get_standard_name(name, choices):
    if not isinstance(name, str): return None
    match, score = process.extractOne(name, choices)
    return match if score > 85 else None

# Normalize Water Data Districts
print("Harmonizing Water Data Districts...")
df_water['district_mapped'] = df_water['district'].apply(lambda x: get_standard_name(x, main_districts))
df_water = df_water.dropna(subset=['district_mapped']).drop(columns=['district']).rename(columns={'district_mapped': 'district'})

# Normalize Vaccine Data Districts
print("Harmonizing Vaccine Data Districts...")
df_vax['district_mapped'] = df_vax['district'].apply(lambda x: get_standard_name(x, main_districts))
df_vax = df_vax.dropna(subset=['district_mapped']).drop(columns=['district']).rename(columns={'district_mapped': 'district'})

# 4. MERGE: WATER (Time-Sensitive Merge)
# We merge on District AND Year because water quality changes over time
print("Merging Historical Water Quality...")
# Group by district/year to handle duplicates (take mean)
df_water_grouped = df_water.groupby(['district', 'year']).agg({'bod': 'mean', 'fecal_coliform': 'mean', 'tds': 'mean'}).reset_index()
df_merged = pd.merge(df_main, df_water_grouped, on=['district', 'year'], how='left')

# Impute missing water data (Forward fill for missing years, then median fill)
df_merged[['bod', 'fecal_coliform', 'tds']] = df_merged.groupby('district')[['bod', 'fecal_coliform', 'tds']].ffill()
df_merged[['bod', 'fecal_coliform', 'tds']] = df_merged[['bod', 'fecal_coliform', 'tds']].fillna(df_merged[['bod', 'fecal_coliform', 'tds']].median())

# 5. MERGE: VACCINES (Static/Snapshot Merge)
# Assuming Vaccine data is a recent snapshot (NFHS-5), we apply it to all years
print("Merging Immunization Profiles...")
df_merged = pd.merge(df_merged, df_vax[['district', 'vax_full', 'vax_measles']], on='district', how='left')
df_merged[['vax_full', 'vax_measles']] = df_merged[['vax_full', 'vax_measles']].fillna(df_merged[['vax_full', 'vax_measles']].median())

# 6. SAVE SUPER DATASET
print(f"--- FUSION COMPLETE. SHAPE: {df_merged.shape} ---")
df_merged.to_csv('SUPER_TRAINING_SET.csv', index=False)
print("Saved to 'SUPER_TRAINING_SET.csv'. Ready for Retraining.")