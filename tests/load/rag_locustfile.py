"""
Locust Load Testing for RAG Pipeline

Load test configuration for testing the RAG pipeline under
various concurrent user loads.
"""

from locust import HttpUser, task, between
import json
import random


class RAGUser(HttpUser):
    """Simulated user for RAG pipeline testing"""
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    # Sample queries for testing
    queries = [
        "How do I configure authentication?",
        "What is GitLab integration?",
        "Explain API rate limiting",
        "How to generate documents?",
        "What are the system requirements?",
        "How do I set up Groq API?",
        "Explain the RAG pipeline",
        "What is LangChain?",
        "How to optimize search performance?",
        "What are the security best practices?",
        "How to handle errors in the system?",
        "Explain token management",
        "What is the document generation process?",
        "How to configure Streamlit?",
        "What are the available AI models?"
    ]
    
    def on_start(self):
        """Initialize user session"""
        # In a real scenario, this might involve authentication
        self.session_id = f"user_{random.randint(1000, 9999)}"
    
    @task(3)
    def query_knowledge_base(self):
        """Simulate knowledge base query (most common operation)"""
        query = random.choice(self.queries)
        
        with self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "session_id": self.session_id,
                "max_tokens": 500
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def search_documents(self):
        """Simulate document search"""
        search_term = random.choice(["auth", "gitlab", "api", "config", "error"])
        
        with self.client.get(
            f"/api/v1/search?q={search_term}&limit=10",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def generate_document(self):
        """Simulate document generation request"""
        with self.client.post(
            "/api/v1/generate",
            json={
                "title": f"Test Document {random.randint(1, 100)}",
                "content": "Generated content based on query results",
                "format": random.choice(["docx", "pdf"])
            },
            catch_response=True
        ) as response:
            if response.status_code in [200, 202]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def check_health(self):
        """Check system health"""
        with self.client.get(
            "/api/v1/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure("System not healthy")
                except:
                    response.failure("Invalid health response")
            else:
                response.failure(f"Got status code {response.status_code}")


class AdvancedRAGUser(RAGUser):
    """Advanced user simulating more complex behaviors"""
    
    @task(2)
    def conversational_query(self):
        """Simulate a conversational query with context"""
        # First query
        initial_query = random.choice(self.queries)
        with self.client.post(
            "/api/v1/query",
            json={
                "query": initial_query,
                "session_id": self.session_id,
                "conversation_id": f"conv_{random.randint(1000, 9999)}"
            }
        ) as response:
            if response.status_code == 200:
                # Follow-up query
                follow_up = "Can you provide more details?"
                self.client.post(
                    "/api/v1/query",
                    json={
                        "query": follow_up,
                        "session_id": self.session_id,
                        "conversation_id": response.json().get("conversation_id")
                    }
                )
    
    @task(1)
    def bulk_search(self):
        """Simulate bulk search operation"""
        search_terms = random.sample(["auth", "gitlab", "api", "config", "error", "token"], 3)
        
        for term in search_terms:
            self.client.get(f"/api/v1/search?q={term}")
    
    @task(1)
    def streaming_query(self):
        """Simulate streaming response query"""
        with self.client.post(
            "/api/v1/query/stream",
            json={
                "query": random.choice(self.queries),
                "session_id": self.session_id,
                "stream": True
            },
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Simulate reading streaming response
                for _ in response.iter_lines():
                    pass
                response.success()
            else:
                response.failure(f"Streaming failed with {response.status_code}")


class StressTestUser(HttpUser):
    """User for stress testing specific endpoints"""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    @task
    def rapid_queries(self):
        """Rapid-fire queries to test rate limiting"""
        for i in range(10):
            self.client.post(
                "/api/v1/query",
                json={
                    "query": f"Quick query {i}",
                    "max_tokens": 50
                },
                name="/api/v1/query [rapid]"
            )


# Custom test scenarios
class MixedLoadTestUser(HttpUser):
    """Realistic mixed load testing"""
    
    wait_time = between(2, 5)
    
    tasks = {
        RAGUser.query_knowledge_base: 50,      # 50% knowledge base queries
        RAGUser.search_documents: 20,          # 20% searches
        RAGUser.check_health: 20,              # 20% health checks
        RAGUser.generate_document: 10          # 10% document generation
    }


# Configuration for different test scenarios
"""
Example Locust commands:

1. Basic load test:
   locust -f tests/load/rag_locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10

2. Stress test:
   locust -f tests/load/rag_locustfile.py --host=http://localhost:8000 --users=1000 --spawn-rate=50 -t 5m

3. Sustained load test:
   locust -f tests/load/rag_locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=5 -t 30m

4. Headless mode with HTML report:
   locust -f tests/load/rag_locustfile.py --host=http://localhost:8000 --users=200 --spawn-rate=20 -t 10m --headless --html=report.html
"""