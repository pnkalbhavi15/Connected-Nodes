import json
import os
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# File paths
FAISS_INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/faiss_metadata.json"
GRAPH_DATA_PATH = "graph_data.json"

# Ensure required files exist
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
    print(" Required files missing! Ensure FAISS index and metadata exist.")
    exit()

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    datasets = json.load(file)

if not datasets:
    print(" No datasets found in metadata file.")
    exit()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Expanded ML topics for better categorization
ml_topics = [
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
    "Deep Learning", "Regression", "Classification", "Clustering",
    "Neural Networks", "CNN", "RNN", "Object Detection", "Pose Estimation",
    "Natural Language Processing", "Computer Vision", "Time Series Analysis",
    "Anomaly Detection", "Recommendation Systems", "Graph Neural Networks",
    "Optimization", "Bayesian Learning", "AutoML", "Federated Learning"
]

def extract_best_matching_category(query):
    """Finds the best ML category for a given query based on similarity."""
    vectorizer = TfidfVectorizer().fit(ml_topics + [query])
    vectors = vectorizer.transform(ml_topics + [query])
    similarity_scores = cosine_similarity(vectors[-1:], vectors[:-1])
    best_match_index = similarity_scores.argmax()
    return ml_topics[best_match_index] if similarity_scores[0][best_match_index] > 0.2 else "Machine Learning"

def extract_keywords(text):
    """Extracts ML-related keywords from metadata using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([text])
    keywords = vectorizer.get_feature_names_out()
    return [kw for kw in keywords if kw.lower() in " ".join(ml_topics).lower()]

def search_datasets(query, top_k=25):  # Increase top_k for more results
    """Search for relevant datasets using FAISS based on user query."""
    query_embedding = np.array([embedding_model.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, top_k)
    return [datasets[i] for i in indices[0] if 0 <= i < len(datasets)]

def build_graph_json(query, best_category, results):
    """Builds a hierarchical graph JSON file."""
    graph_data = {"name": "ML Research", "children": []}
    category_node = {"name": best_category, "children": []}

    # Extract and cluster keywords
    keyword_clusters = defaultdict(list)
    for paper in results:
        metadata_text = paper.get("metadata", "")
        keywords = extract_keywords(metadata_text)
        for keyword in keywords:
            keyword_clusters[keyword].append(paper)

    for keyword, papers in keyword_clusters.items():
        keyword_node = {"name": keyword, "children": []}
        for paper in papers:
            paper_node = {"name": paper["name"], "link": paper["link"]}
            keyword_node["children"].append(paper_node)
        category_node["children"].append(keyword_node)

    graph_data["children"].append(category_node)

    with open(GRAPH_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)

    print(f" Graph data updated in {GRAPH_DATA_PATH}")

# User input
query = input("\n Enter your research query: ").strip()
best_category = extract_best_matching_category(query)
results = search_datasets(query)

# Update graph JSON
build_graph_json(query, best_category, results)

# Display results
if results:
    print("\n🔹 Top Matching Datasets:")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['name']}")
        print(f"    Topic: {result['topic']}")
        print(f"    Link: {result['link']}")
        print(f"   Metadata: {result['metadata']}\n")
else:
    print(" No matching datasets found.")

print(" Graph JSON file generated. Open `ml_graph.html` to visualize it.")
