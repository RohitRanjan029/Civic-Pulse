import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

class GraphManager:
    def __init__(self):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(URI, auth=AUTH)
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j AuraDB")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def push_simulation_result(self, intel):
        if not self.driver: return

        query = """
        MERGE (d:District {name: $dist_name})
        SET d.state = $state, d.population = $pop, d.beds = $beds
        
        MERGE (dis:Disease {name: $disease})
        
        CREATE (e:Event {
            type: 'AI_PREDICTION',
            timestamp: datetime(),
            predicted_cases: $cases,
            status: $status,
            insight: $insight
        })
        
        MERGE (d)-[:EXPERIENCING]->(e)
        MERGE (e)-[:IS_OUTBREAK_OF]->(dis)
        
        WITH d, e
        CALL apoc.do.when($is_collapse, 
            'MERGE (r:Risk {name: "Hospital System Collapse"}) MERGE (e)-[:TRIGGERS]->(r) MERGE (d)-[:AT_RISK_OF]->(r)',
            '', {d:d, e:e}) YIELD value AS v1
            
        CALL apoc.do.when($is_manpower_issue, 
            'MERGE (m:Risk {name: "Critical Doctor Shortage"}) MERGE (e)-[:EXACERBATED_BY]->(m) MERGE (d)-[:HAS_WEAKNESS]->(m)',
            '', {d:d, e:e}) YIELD value AS v2
        """
        
        safe_query = """
        MERGE (d:District {name: $dist_name})
        SET d.state = $state, d.population = $pop, d.beds = $beds
        MERGE (dis:Disease {name: $disease})
        CREATE (e:Event {type: 'AI_PREDICTION', timestamp: datetime(), predicted_cases: $cases, status: $status})
        MERGE (d)-[:EXPERIENCING]->(e)
        MERGE (e)-[:IS_OUTBREAK_OF]->(dis)
        """

        is_collapse = "COLLAPSE" in intel['status']
        is_manpower = "DECEPTIVE" in intel.get('governance_insight', '') or "COLLAPSE" in intel.get('manpower_status', '')

        with self.driver.session() as session:
            try:
                session.run(query, 
                    dist_name=intel['district'],
                    state=intel.get('state', 'Unknown'),
                    pop=intel.get('population', 0),
                    beds=intel['capacity_status']['beds_available'],
                    disease=intel['disease'],
                    cases=intel['prediction'],
                    status=intel['status'],
                    insight=intel.get('governance_insight', 'None'),
                    is_collapse=is_collapse,
                    is_manpower_issue=is_manpower
                )
            except:
                session.run(safe_query, 
                    dist_name=intel['district'],
                    state=intel.get('state', 'Unknown'),
                    pop=intel.get('population', 0),
                    beds=intel['capacity_status']['beds_available'],
                    disease=intel['disease'],
                    cases=intel['prediction'],
                    status=intel['status']
                )

    def push_news_intel(self, news_data):
        if not self.driver: return

        query = """
        MERGE (d:District {name: $dist_name})
        WITH d
        FOREACH (topic IN $topics | 
            MERGE (t:TrendingTopic {name: topic})
            MERGE (d)-[:DISCUSSING]->(t)
        )
        WITH d
        FOREACH (problem IN $problems |
            MERGE (p:HiddenProblem {category: problem})
            MERGE (d)-[:SUFFERING_FROM]->(p)
        )
        """
        with self.driver.session() as session:
            session.run(query, 
                dist_name=news_data['district'],
                topics=news_data['raw_entities_found'],
                problems=news_data['hidden_problems_detected']
            )

    def get_district_ontology(self, district_name):
        if not self.driver: return {"error": "Database disconnected"}

        # CYPER SYNTAX FIX: Using List Comprehensions instead of CASE inside aggregations
        query = """
        MATCH (d:District {name: $dist_name})
        
        OPTIONAL MATCH (d)-[:SUFFERING_FROM]->(p:HiddenProblem)
        WITH d, collect(DISTINCT p.category) as hidden_problems
        
        OPTIONAL MATCH (d)-[:DISCUSSING]->(t:TrendingTopic)
        WITH d, hidden_problems, collect(DISTINCT t.name) as trending_topics
        
        OPTIONAL MATCH (d)-[:EXPERIENCING]->(e:Event)-[:IS_OUTBREAK_OF]->(dis:Disease)
        WITH d, hidden_problems, trending_topics, 
             collect(DISTINCT {disease: dis.name, cases: e.predicted_cases, status: e.status}) as raw_threats
             
        WITH d, hidden_problems, trending_topics, 
             [threat IN raw_threats WHERE threat.cases IS NOT NULL] as active_threats
        
        OPTIONAL MATCH (d)-[:AT_RISK_OF|HAS_WEAKNESS]->(r:Risk)
        WITH d, hidden_problems, trending_topics, active_threats, collect(DISTINCT r.name) as systemic_risks
        
        RETURN {
            district: d.name,
            state: d.state,
            population: d.population,
            beds: d.beds,
            hidden_problems: hidden_problems,
            trending_topics: trending_topics,
            ml_predictions: active_threats,
            systemic_risks: systemic_risks
        } as ontology
        """
        with self.driver.session() as session:
            result = session.run(query, dist_name=district_name)
            record = result.single()
            if record:
                return record["ontology"]
            return {"message": "No data found for this district in the Knowledge Graph."}

graph_db = GraphManager()