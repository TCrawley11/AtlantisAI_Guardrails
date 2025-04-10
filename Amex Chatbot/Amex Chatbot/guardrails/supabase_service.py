import os
from supabase import create_client, Client
from guardrails import Checker

# Replace these with your actual Supabase project URL and anon key.
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your_supabase_anon_key")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_embedding(query: str):
    # Get the embedding from the Checker.
    embedding = Checker.get_embeddings(query)
    
    # Data format: the embedding column in your table is a vector type.
    # Depending on the Supabase/PostgREST version, sending the embedding as a list of floats
    # is often acceptable.
    data = {"query": query, "embedding": embedding}
    
    # Insert the embedding into the table.
    response = supabase.table("embeddings").insert(data).execute()
    print("Insert response:", response)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    # Optionally, print embeddings for debugging.
    print("Embeddings:", Checker.get_embeddings(user_query))
    store_embedding(user_query)
