import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# STEP 1: Load JSON Data
with open("data\datasets.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# Extract required fields
documents = []
for paper in papers:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    keywords = ", ".join(paper.get("keywords", []))  # Convert list to string
    full_text = paper.get("full_text", "")  # Optional

    # Combine fields into one text block
    content = f"Title: {title}\nKeywords: {keywords}\nAbstract: {abstract}\n{full_text}"
    
    documents.append({"id": paper.get("id", None), "text": content})

# STEP 2: Generate Text Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_texts = [doc["text"] for doc in documents]
document_embeddings = embedding_model.encode(document_texts, convert_to_tensor=True)

# STEP 3: Store Embeddings in FAISS
embeddings_np = document_embeddings.cpu().detach().numpy()
embedding_dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance for similarity
index.add(embeddings_np)  # Add embeddings

# Save FAISS index
faiss.write_index(index, "data/faiss_index.bin")

# Save document metadata
with open("data\datasets.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=4)

print("✅ Preprocessing complete! FAISS index & metadata saved.")
