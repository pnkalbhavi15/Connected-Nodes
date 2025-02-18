import json
import os
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load dataset
DATA_PATH = "data/datasets.json"
with open(DATA_PATH, "r", encoding="utf-8") as file:
    datasets = json.load(file)

# Extract text for embedding
documents = [f"{d['name']} {d['topic']} {d['metadata']}" for d in datasets]

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generate embeddings
print("🔄 Generating embeddings...")
embeddings = embedding_model.embed_documents(documents)
embeddings = np.array(embeddings, dtype=np.float32)

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/faiss_index.bin")

# Save dataset mapping
metadata_path = "data/faiss_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(datasets, f, indent=4)

print(f"✅ FAISS index and metadata saved in 'data/faiss_index.bin' and '{metadata_path}'")
