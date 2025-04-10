import os, json, torch, time, re, torch, random, time
import numpy as np
import concurrent.futures
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Document, GPTVectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

from guardrails import check, connect, observe

"""
Run from parent folder (amex chatbot) with python -m Code.chatbot_ai
"""

# Set up for TinyLlama LLM model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tinyllama_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tinyllama_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tinyllama_pipeline = pipeline(
    "text-generation",
    model=tinyllama_model,
    tokenizer=tinyllama_tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=100
)

# Reads rewards data from JSON
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'amex_rewards.json')
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["rewards"]

# Converts data into structured documents
def create_documents(data):
    documents = [
        Document(
            text=f"Program: {item['program']}\nConversion Rate: {item['conversion_rate']}\nDescription: {item['description']}",
            metadata={"program": item["program"]}
        ) 
        for item in data
    ]
    return documents

def load_tinyllama_llm():
    return HuggingFaceLLM(
        model=tinyllama_model,
        tokenizer=tinyllama_tokenizer,
        context_window=2048  
    )

# Creates or loads a vector store for fast retrieval.
def build_or_load_index(documents):
    INDEX_DIR = "./index_store"

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    # Load TinyLlama 
    Settings.llm = load_tinyllama_llm()

    # Load existing index
    if os.path.exists(INDEX_DIR):   
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR) 
        return load_index_from_storage(storage_context)

    # Build new index
    else:   
        index = GPTVectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        return index

# A lightweight embedding model for semantic filtering
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Checks if query has numbers
def contains_numbers(text):
    return bool(re.search(r'\d+', text))

# Retrieves relevant results from the index   
def retrieve_data(index, query):
    max_tokens = get_dynamic_max_tokens 

    # Query Engine Setup
    query_engine = index.as_query_engine(
        similarity_top_k=3,  
        max_new_tokens=max_tokens,
        stop=["End of answer"] 
    )
    
    response = query_engine.query(query)
    retrieved_texts = response.response.split("\n")

    # Priority 1: Conversion Rates Instant Return
    if "conversion rate" in query.lower() or "points to dollars" in query.lower():
        conversion_rates = [line for line in retrieved_texts if contains_numbers(line)]
        if conversion_rates:
            return conversion_rates[0]

    # Priority 2: List Handling (Partial + Full)
    list_items = [line for line in retrieved_texts if line.strip().startswith(("•", "-", "1.", "2.", "3."))]
    if list_items:
        partial = list_items[:5]  
        remainder = list_items[5:15]  
        return "\n".join(partial + (["...more results available"] if len(remainder) > 0 else remainder))

    # Priority 3: Amex Keyword Results (Fallback)
    keywords = ["amex", "membership rewards", "points", "redeem", "air canada", "travel"]
    valid_results = [
        text for text in retrieved_texts
        if any(keyword in text.lower() for keyword in keywords) and len(text.split()) > 3
    ]

    return valid_results[0] if valid_results else "No relevant information found."

# Filters irrelevant sentences
def filter_irrelevant_sentences(response, retrieved_text):
    # Split response into sentences
    sentences = response.split(". ")
    filtered_sentences = []

    # Only keep sentences that have high similarity with retrieved data
    for sentence in sentences:
        similarity = fuzz.token_set_ratio(sentence, retrieved_text)
        if similarity > 60:  
            filtered_sentences.append(sentence)

    return ". ".join(filtered_sentences).strip()

# Dynamically adjust tokens based on query
def get_dynamic_max_tokens(query):
    word_count = len(query.split())

    if word_count <= 5: return 100  
    elif word_count <= 12: return 200 
    else: return 450

# Generates AI response
def generate_response(pipe, retrieved_text, user_query):
    # Fallback for no relevant info
    if "No relevant information found" in retrieved_text:
        return "I'm sorry, but I couldn't find any reliable information on that topic."

    # Short numeric responses handled directly
    if contains_numbers(retrieved_text) and len(retrieved_text.split()) <= 10:
        return retrieved_text

    # Structured prompt for generation
    prompt = f"""
    You are an AI assistant specializing in American Express rewards programs.
    Answer the user's question concisely using only the retrieved information.

    --- Information Retrieved ---
    {retrieved_text}

    --- Question ---
    {user_query}

    --- Instructions ---
    - Generate a direct, fact-based answer based solely on the retrieved data.
    - Avoid adding speculative details or assumptions not in the data.
    - Keep responses concise, limiting to essential facts only.
    - Do not repeat the information exactly; rephrase concisely while preserving meaning.

    ### Answer:
    """

    output = pipe(
        prompt,
        max_new_tokens = get_dynamic_max_tokens(user_query), 
        do_sample = True,
        temperature = 0.3,  
        top_p = 0.8
    )

    ai_response = output[0]["generated_text"].split("Answer:")[-1].strip()
    final_response = filter_irrelevant_sentences(ai_response, retrieved_text)
    return final_response or "I'm sorry, but I couldn't find any reliable information on that topic."

if __name__ == "__main__":
    # Load & index data
    data = load_data()
    documents = create_documents(data)
    index = build_or_load_index(documents)

    print("\n🔹 Amex Rewards Assistant Chatbot 🔹\n")

    while True:
        user_input = input("💬: ").strip()

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("🤖 Thinking...", flush=True)
        retrieved_text = retrieve_data(index, user_input)
        with torch.inference_mode(): ai_response = generate_response(tinyllama_pipeline, retrieved_text, user_input)
        print(f"\n{ai_response}\n")
        
        ## Testing guardrails ##
        """
        This code currently breaks the process because it prints too much too fast 
        checker = check()
        print(checker.print_embeddings(user_input))
        """
       