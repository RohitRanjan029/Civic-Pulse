import os
import shutil
import glob
import re

base_dir = r"c:\Users\ayush\Desktop\Inoovate"
proj_dir = os.path.join(base_dir, "Proj")
hack_dir = os.path.join(base_dir, "Inovate_Hackathon")

# Create structure
dirs = [
    os.path.join(hack_dir, "ai_engine", "data"),
    os.path.join(hack_dir, "ai_engine", "models"),
    os.path.join(hack_dir, "client"),
    os.path.join(hack_dir, "knowledge_graph")
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Move Data (.csv)
for src in glob.glob(os.path.join(proj_dir, "*.csv")):
    shutil.move(src, os.path.join(hack_dir, "ai_engine", "data", os.path.basename(src)))

# Move Models (.json, .npy)
for src in glob.glob(os.path.join(proj_dir, "*.json")):
    shutil.move(src, os.path.join(hack_dir, "ai_engine", "models", os.path.basename(src)))
for src in glob.glob(os.path.join(proj_dir, "*.npy")):
    shutil.move(src, os.path.join(hack_dir, "ai_engine", "models", os.path.basename(src)))

# Process logic_engine.py
omega_src = os.path.join(proj_dir, "final_omega_engine.py")
logic_dest = os.path.join(hack_dir, "ai_engine", "logic_engine.py")

with open(omega_src, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix paths
content = content.replace("MODEL_FILE = 'sovereign_brain_perfected.json'", "MODEL_FILE = 'models/sovereign_brain_perfected.json'")
content = content.replace("ENCODER_FILE = 'disease_encoder_v2.npy'", "ENCODER_FILE = 'models/disease_encoder_v2.npy'")
content = content.replace("POPULATION_FILE = 'System_Collapse_Baseline.csv'", "POPULATION_FILE = 'data/System_Collapse_Baseline.csv'")
content = content.replace("WATER_FILE = 'Cleaned_Historical_Water_Quality.csv'", "WATER_FILE = 'data/Cleaned_Historical_Water_Quality.csv'")
content = content.replace("VAX_FILE = 'Cleaned_Immunization_History.csv'", "VAX_FILE = 'data/Cleaned_Immunization_History.csv'")
content = content.replace("EPICLIM_FILE = 'Cleaned_EpiClim_Data.csv'", "EPICLIM_FILE = 'data/Cleaned_EpiClim_Data.csv'")
content = content.replace("MANPOWER_FILE = 'Cleaned_Manpower_State_Stats.csv'", "MANPOWER_FILE = 'data/Cleaned_Manpower_State_Stats.csv'")
content = content.replace("INFRA_FILE = 'MASTER_Analytical_Base_Table.csv'", "INFRA_FILE = 'data/MASTER_Analytical_Base_Table.csv'")
content = content.replace("with open('final_dashboard_payload.json', 'w')", "with open('models/final_dashboard_payload.json', 'w')")

# Fix main guard so when logic_engine is imported, it doesn't run the hardcoded simulation.
content = content.replace("\nresults = []", "\nif __name__ == '__main__':\n    results = []")
content = content.replace("\nprint(\"\\n[SYSTEM]", "\n    print(\"\\n[SYSTEM]")
content = content.replace("\nresults.append", "\n    results.append")
content = content.replace("\nwith open", "\n    with open")
content = content.replace("\n    json.dump", "\n        json.dump")
content = content.replace("\nprint(\"\\n[COMPLETE]", "\n    print(\"\\n[COMPLETE]")


with open(logic_dest, 'w', encoding='utf-8') as f:
    f.write(content)

# Create main.py for FastAPI setup
main_dest = os.path.join(hack_dir, "ai_engine", "main.py")
main_code = '''from fastapi import FastAPI
from logic_engine import run_omega_simulation

app = FastAPI(title="Omega Engine API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Omega Engine API"}

@app.get("/simulate")
def simulate(district: str, month: int, rain: float, temp: float, lai: float, disease: str):
    """
    Run the Omega Simulation for a given district and conditions.
    """
    result = run_omega_simulation(district, month, rain, temp, lai, disease)
    if result is None:
        return {"status": "error", "message": "Simulation failed. Check input parameters or district name."}
    return {"status": "success", "data": result}
'''
with open(main_dest, 'w', encoding='utf-8') as f:
    f.write(main_code)

print("Restructuring Complete")
