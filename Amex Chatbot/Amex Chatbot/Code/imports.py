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