import json
import os
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File paths
FAISS_INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/faiss_metadata.json"
GRAPH_DATA_PATH = "graph_data.json"

# Ensure required files exist
if not os.path.exists(FAISS_INDEX_PATH):
    print(" FAISS index file not found! Run `build_faiss.py` first.")
    exit()

if not os.path.exists(METADATA_PATH):
    print(" Metadata file not found! Ensure `data/faiss_metadata.json` exists.")
    exit()

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    datasets = json.load(file)

if not datasets:
    print(" No datasets found in metadata file.")
    exit()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Predefined ML-related categories
ml_topics = [
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
    "Deep Learning", "Regression", "Classification", "Clustering",
    "Neural Networks", "Decision Trees", "Random Forest", "Support Vector Machines",
    "CNN", "RNN"
]

def extract_best_matching_category(query):
    """Finds the best ML category for a given query based on similarity."""
    vectorizer = TfidfVectorizer().fit(ml_topics + [query])
    vectors = vectorizer.transform(ml_topics + [query])
    similarity_scores = cosine_similarity(vectors[-1:], vectors[:-1])

    best_match_index = similarity_scores.argmax()
    best_match_score = similarity_scores[0][best_match_index]

    return ml_topics[best_match_index] if best_match_score > 0.3 else "Machine Learning"

def search_datasets(query, top_k=10):
    """Search for relevant datasets using FAISS based on user query."""
    if not query.strip():
        print(" Query cannot be empty. Please enter a valid search term.")
        return []

    query_embedding = np.array([embedding_model.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, top_k)

    return [datasets[i] for i in indices[0] if 0 <= i < len(datasets)]

def update_graph_json(query, best_category, results):
    """Generates a properly structured hierarchical JSON file for D3.js visualization."""
    graph_data = {"name": "ML Research", "children": []}

    # Check if the best category exists, else create it
    category_node = next((c for c in graph_data["children"] if c["name"] == best_category), None)
    if not category_node:
        category_node = {"name": best_category, "children": []}
        graph_data["children"].append(category_node)

    # Add query as a subtopic under the best category
    query_node = {"name": query, "children": []}
    category_node["children"].append(query_node)

    # Add paper nodes under the query
    for paper in results:
        paper_node = {"name": paper["name"], "children": []}

        # Parse metadata correctly
        metadata = paper.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Add metadata as child nodes
        for key, value in metadata.items():
            paper_node["children"].append({"name": f"{key}: {value}"})

        query_node["children"].append(paper_node)

    # Save JSON
    with open(GRAPH_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)

    print(f" Graph data updated in {GRAPH_DATA_PATH}")

# User input
query = input("\n Enter your research query: ").strip()
best_category = extract_best_matching_category(query)
results = search_datasets(query)

# Update graph JSON with new data
update_graph_json(query, best_category, results)

# Display results
if results:
    print("\n🔹 Top Matching Datasets:")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['name']}")
        print(f"     Topic: {result['topic']}")
        print(f"     Link: {result['link']}")
        print(f"     Metadata: {result['metadata']}\n")
else:
    print(" No matching datasets found.")
