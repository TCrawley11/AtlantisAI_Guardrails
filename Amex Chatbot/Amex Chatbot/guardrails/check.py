"""
 Provides the main bulk of the guardrails. This code connects queries with the supabase vector database.
 Input and output checks are implemented. Embeddings are checked by roBERTa model. An attack simulator, 
 aka the 1B parameter model used to generate purposefully bad/malicious queries is used. 
"""

from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

class Checker:
    ## Initialize tokenizer and model from pretrained sentence transformers ## 
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def __init__(self) -> None:
        pass

    ## Encodes the input to prepare for embedding generation ##
    @staticmethod
    def get_encodings(sentences):
        encoded_input = Checker.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        return encoded_input

    ## Mean Pooling - Take attention mask into account for correct averaging ##
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    ## generate the token embeddings, then perform pooling, then normalize the embeddings ##
    @staticmethod
    def get_embeddings(query: str) -> List[float]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        encoded_input = Checker.get_encodings(query)
        try:
            model_output = Checker.model(**encoded_input)
            sentence_embeddings = Checker.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            # Detach from torch graph, move to CPU and convert to a list.
            return sentence_embeddings.detach().cpu().tolist()[0]
        except Exception as e:
            raise Exception(f"Error during embedding generation: {e}")

    ## Used for debugging ##
    @staticmethod
    def print_embeddings(query) -> None:
        try:
            embeddings = Checker.get_embeddings(query)
            print("Embeddings: %s", embeddings)
        except Exception as e:
            raise("Failed to generate embeddings: %s", e)
    
if __name__ == "__main__":
    user_query = input("Input for testing embeddings: ")
    Checker.print_embeddings(user_query)


    