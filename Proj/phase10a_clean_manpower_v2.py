import pandas as pd
import numpy as np

print("--- PHASE 10a: MANPOWER DATA SANITIZATION (V2 - SMART COLUMN FIX) ---")

FILES = {
    "PHC_Docs": "raw_phc_doctors.csv",
    "Surgeons": "raw_chc_surgeons.csv",
    "OBGYN": "raw_chc_obgyn.csv",
    "Physicians": "raw_chc_physicians.csv",
    "Paediatricians": "raw_chc_paediatricians.csv",
    "Anaesthetists": "raw_chc_anaesthetists.csv"
}

def clean_value(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    if s in ["N App", "NA", "-", "Nil", "nan"]: return 0
    s = s.replace('*', '').replace(',', '').replace('+', '')
    try:
        return int(float(s))
    except ValueError:
        return 0

master_data = []

for category, filename in FILES.items():
    try:
        df = pd.read_csv(filename)
        # Normalize headers
        df.columns = [c.strip() for c in df.columns]
        
        # SMART SELECTOR: Find the State column
        # Look for columns containing "State" or "UT"
        col_state = next((c for c in df.columns if "State" in c or "UT" in c), None)
        
        # Fallback: If not found, check if column 1 looks like text (often column 0 is S.No)
        if not col_state:
            # Check if column 1 contains strings like "Andhra"
            if df.shape[1] > 1 and df.iloc[0, 1].strip().isalpha():
                col_state = df.columns[1]
            else:
                col_state = df.columns[0] # Desperate fallback

        # Find In Position
        col_pos = next((c for c in df.columns if "Position" in c), None)
        col_vac = next((c for c in df.columns if "Vacant" in c), None)
        col_sanc = next((c for c in df.columns if "Sanctioned" in c), None)
        
        if not col_pos:
            print(f"⚠️ SKIPPING {category}: 'In Position' column missing.")
            continue

        print(f"Processing {category}... (State Col: '{col_state}')")

        for _, row in df.iterrows():
            state_name = str(row[col_state]).strip()
            # Cleanup unwanted rows
            if state_name.lower() in ["nan", "all india", "total", "nan"]: continue
            if state_name.replace('.', '').isdigit(): continue # Skip row if state name is a number
            
            in_position = clean_value(row[col_pos])
            vacant = clean_value(row[col_vac]) if col_vac else 0
            sanctioned = clean_value(row[col_sanc]) if col_sanc else (in_position + vacant)
            
            shortfall_pct = 0.0
            if sanctioned > 0:
                shortfall_pct = round((vacant / sanctioned) * 100, 1)
            
            master_data.append({
                "State": state_name,
                "Category": category,
                "In_Position": in_position,
                "Shortfall_Pct": shortfall_pct
            })
            
    except FileNotFoundError:
        print(f"❌ ERROR: File {filename} not found.")

if not master_data:
    print("CRITICAL: No data processed. Check CSV headers.")
    exit()

df_master = pd.DataFrame(master_data)

# Pivot table
df_pivot = df_master.pivot_table(
    index='State', 
    columns='Category', 
    values=['In_Position', 'Shortfall_Pct'],
    aggfunc='sum'
).reset_index()

# Flatten headers
df_pivot.columns = [f"{c[1]}_{c[0]}" if c[1] else c[0] for c in df_pivot.columns]

# Calculate Totals
pos_cols = [c for c in df_pivot.columns if "_In_Position" in c]
df_pivot['Total_Doctors_Available'] = df_pivot[pos_cols].sum(axis=1)

print(f"\n[SUCCESS] Processed {len(df_pivot)} States.")
print(df_pivot[['State', 'Total_Doctors_Available']].head())

df_pivot.to_csv("Cleaned_Manpower_State_Stats.csv", index=False)
print("Saved clean file.")