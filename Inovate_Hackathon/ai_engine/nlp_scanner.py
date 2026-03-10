import spacy
import requests
import os
import time
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

print("[SYSTEM] Initializing Real-Time NLP City Scanner...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

NEWS_CACHE = {}
CACHE_EXPIRY_SECONDS = 3600 

PROBLEM_ONTOLOGY = {
    "Water Toxicity":["hard water", "stomach pain", "dirty water", "brown water", "diarrhea", "cholera", "pollution", "contaminated"],
    "Infrastructure Failure":["traveling", "no doctor", "transferred", "out of meds", "damaged road", "hospital full", "no beds", "strike"],
    "Air Quality Crisis": ["smog", "breathing", "cough", "smoke", "aqi", "asthma"],
    "Sanitation Collapse":["garbage", "waterlogging", "mosquitoes", "overflowing", "dengue", "drainage"]
}

def fetch_live_news(district_name: str) -> list:
    current_time = time.time()

    if district_name in NEWS_CACHE:
        cached_data = NEWS_CACHE[district_name]
        if (current_time - cached_data["timestamp"]) < CACHE_EXPIRY_SECONDS:
            print(f"⚡ [CACHE HIT] Loading news for {district_name} from memory (Saves API call).")
            return cached_data["articles"]

    print(f"🌐 [API CALL] Fetching live internet news for {district_name}...")
    
    if not NEWS_API_KEY:
        print("⚠️ ERROR: NEWS_API_KEY is missing from .env file.")
        return[]

    query = f'"{district_name}" AND (hospital OR water OR pollution OR disease OR doctor OR road OR infrastructure OR protest)'
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=10&apiKey={NEWS_API_KEY}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        articles_text =[]
        if data.get("status") == "ok":
            for article in data.get("articles",[]):
                title = str(article.get("title") or "")
                desc = str(article.get("description") or "")
                if title and desc:
                    articles_text.append(f"{title}. {desc}")
        
        NEWS_CACHE[district_name] = {
            "timestamp": current_time,
            "articles": articles_text
        }
        
        return articles_text
    
    except Exception as e:
        print(f"❌ API Error: {e}")
        return[]

def scan_city_news(district_name: str):
    articles = fetch_live_news(district_name)
    
    if not articles:
        return {
            "district": district_name,
            "scanned_articles": 0,
            "raw_entities_found": [],
            "hidden_problems_detected":["No recent data found in news."]
        }

    detected_problems = set()
    extracted_entities =[]

    for text in articles:
        doc = nlp(text.lower())
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 4 and chunk.root.pos_ == "NOUN":
                extracted_entities.append(chunk.text)

        for token in doc:
            for problem_category, keywords in PROBLEM_ONTOLOGY.items():
                if token.lemma_ in [kw.split()[-1] for kw in keywords] or token.text in [kw for kw in keywords]:
                    detected_problems.add(problem_category)

    entity_counts = [item[0] for item in Counter(extracted_entities).most_common(5)]

    return {
        "district": district_name,
        "scanned_articles": len(articles),
        "raw_entities_found": entity_counts,
        "hidden_problems_detected": list(detected_problems) if detected_problems else["No critical governance issues detected in current news cycle."]
    }

if __name__ == "__main__":
    print("\n--- INITIATING LIVE NLP SCAN ---")
    result1 = scan_city_news("Delhi")
    print(f"\nScan 1 Results: {result1['hidden_problems_detected']}")
    print(f"Trending Topics: {result1['raw_entities_found']}")
    
    print("\n--- TESTING CACHE ---")
    result2 = scan_city_news("Delhi")