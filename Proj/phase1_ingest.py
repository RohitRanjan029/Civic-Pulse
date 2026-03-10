import pandas as pd
import numpy as np

# --- CONFIGURATION ---
HAZARD_FILE = '3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv'
VULNERABILITY_FILE = 'NFHS_5_India_Districts_Factsheet_Data.csv'
CAPACITY_FILE = 'RS_Session_257_AU_282_1.csv'

print("--- PHASE 1: DATA INGESTION ---")

# 1. LOAD & TRANSFORM HAZARD DATA (Live Sensor Data)
print(f"Loading Hazard Data: {HAZARD_FILE}...")
df_hazard_raw = pd.read_csv(HAZARD_FILE)

# Pivot the table: We need one row per City, with pollutants as columns
# aggregating by max value to catch the worst-case scenario in a city
df_hazard = df_hazard_raw.pivot_table(
    index=['state', 'city'], 
    columns='pollutant_id', 
    values='pollutant_avg', 
    aggfunc='max'
).reset_index()

# Clean up column names
df_hazard.columns.name = None
print(f"Hazard Data Shape after Pivot: {df_hazard.shape}")
print("Hazard Columns:", df_hazard.columns.tolist())


# 2. LOAD VULNERABILITY DATA (Demographics/Health)
print(f"\nLoading Vulnerability Data: {VULNERABILITY_FILE}...")
df_vuln = pd.read_csv(VULNERABILITY_FILE)

# Clean column headers (strip spaces, newlines, and annoying characters)
df_vuln.columns = [c.strip().replace('\n', ' ').replace('  ', ' ') for c in df_vuln.columns]
print(f"Vulnerability Data Shape: {df_vuln.shape}")
# Let's verify we have the critical 'District Names' and 'State/UT' columns
print("First 5 Vulnerability Columns:", df_vuln.columns.tolist()[:5])


# 3. LOAD CAPACITY DATA (Infrastructure)
print(f"\nLoading Capacity Data: {CAPACITY_FILE}...")
df_cap = pd.read_csv(CAPACITY_FILE)
df_cap.columns = [c.strip() for c in df_cap.columns]
print(f"Capacity Data Shape: {df_cap.shape}")


# 4. PRELIMINARY MATCH ANALYSIS
# We need to link these datasets. The common link is 'State' and 'District/City'.
# Let's see if the State names match between Hazard and Capacity.

hazard_states = set(df_hazard['state'].unique())
cap_states = set(df_cap['State/UT'].unique())

print("\n--- MERGE INTEGRITY CHECK ---")
common_states = hazard_states.intersection(cap_states)
print(f"Total Unique Hazard States: {len(hazard_states)}")
print(f"Total Unique Capacity States: {len(cap_states)}")
print(f"Matching States: {len(common_states)}")

# List mismatches (Critical for Phase 2)
print("\nStates in Hazard Data NOT found in Capacity Data (Potential Mismatches):")
print(hazard_states - cap_states)

print("\nStates in Capacity Data NOT found in Hazard Data:")
print(cap_states - hazard_states)

print("\n--- PHASE 1 COMPLETE. REPORT OUTPUT ABOVE ---")