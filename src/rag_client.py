import requests

class RAGClient:
    """Client to interact with RAG endpoint."""
    def __init__(self, endpoint_url: str, api_key: str = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def health_check(self) -> bool:
        """Check the health of the RAG endpoint."""
        try:
            response = requests.get(f"{self.endpoint_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    # Function to call RAG API 
    def retrieve(self, question: str, k: int = 20):
        """
        Sends a question to the RAG API and returns the response.
        """
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "rag_method": "standard",
            "k": k,
            "question": question
        }
        
        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        # Raise an exception if the call failed
        response.raise_for_status()

        data = response.json()
        # Extract reranked results safely
        results = data.get("reranked_results", [])
        return results
