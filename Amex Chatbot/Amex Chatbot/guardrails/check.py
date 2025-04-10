"""
 Provides the main bulk of the guardrails. This code connects queries with the supabase vector database.
 Input and output checks are implemented. Embeddings are checked by roBERTa model. An attack simulator, 
 aka the 1B parameter model used to generate purposefully bad/malicious queries is used. 
"""

from typing import List

from sentence_transformers import SentenceTransformer

class Checker:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __init__(self) -> None:
        pass

    ## generate the embeddings with the defined model ##
    def get_embeddings(self, query) -> List[float]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        try:
            return Checker.model.encode(query)
        except Exception as e:
            raise("Error during embedding generation: %s", e)
    
    ## Used for debugging ##
    def print_embeddings(self, query) -> None:
        try:
            embeddings = self.get_embeddings(query)
            print("Embeddings: %s", embeddings)
        except Exception as e:
            raise("Failed to generate embeddings: %s", e)

    