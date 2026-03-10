import requests
import json
import time

BASE_URL = "http://localhost:8001"

print("--- TESTING FULL ENTERPRISE BACKEND PIPELINE ---")

print("\n1. Triggering NLP Scanner for Dehradun...")
news_res = requests.get(f"{BASE_URL}/news/Dehradun")
print(f"Status: {news_res.status_code}")

time.sleep(1)

print("\n2. Triggering ML AI Prediction for Dehradun...")
payload = {
    "district": "Dehradun",
    "month": 8,
    "rain": 400.0,
    "temp": 30.0,
    "lai": 4.0,
    "disease": "Malaria"
}
predict_res = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"Status: {predict_res.status_code}")

time.sleep(1)

print("\n3. Fetching Final Ontology Dashboard from Neo4j...")
dashboard_res = requests.get(f"{BASE_URL}/dashboard/Dehradun")
print(json.dumps(dashboard_res.json(), indent=2))