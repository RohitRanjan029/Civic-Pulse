from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logic_engine 
from graph_db import graph_db 
import nlp_scanner 

app = FastAPI(title="Sovereign Risk Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    district: str
    month: int
    rain: float
    temp: float
    lai: float
    disease: str

@app.get("/")
def home():
    return {"status": "System Online", "module": "Sovereign AI Engine"}

@app.post("/predict")
def predict_outbreak(req: SimulationRequest):
    try:
        print(f"🔄 Receiving Simulation Request: {req.district} ({req.disease})")
        
        result = logic_engine.run_omega_simulation(
            dist_name=req.district,
            month=req.month,
            rain=req.rain,
            temp=req.temp,
            lai=req.lai,
            disease_name=req.disease
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="District/Disease not found in Knowledge Base")
            
        if "error" in result:
             raise HTTPException(status_code=404, detail=result["error"])

        try:
            graph_db.push_simulation_result(result)
        except Exception as e:
            print(f"⚠️ Graph Sync Warning: {e}")

        return result
    
    except Exception as e:
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/{district}")
def get_city_news(district: str):
    try:
        print(f"📰 Scanning News for: {district}")
        result = nlp_scanner.scan_city_news(district)
        
        try:
            if result.get("scanned_articles", 0) > 0:
                print(f"🧠 Syncing NLP Insights to Neo4j...")
                graph_db.push_news_intel(result)
        except Exception as e:
            print(f"⚠️ Graph NLP Sync Warning: {e}")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ADDITION: THE DASHBOARD ENDPOINT ---
@app.get("/dashboard/{district}")
def get_dashboard_data(district: str):
    """
    The ultimate API for the Frontend Map. 
    Retrieves the fused Ontology (News + ML + Infra) directly from Neo4j.
    """
    try:
        print(f"🌍 Fetching Complete Dashboard Ontology for: {district}")
        ontology = graph_db.get_district_ontology(district)
        return ontology
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)