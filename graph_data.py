import json
import torch
import random
import pandas as pd
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Load SBERT Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define Machine Learning Dictionary
ml_terms = [
    "deep learning", "neural network", "gradient descent", "reinforcement learning",
    "support vector machine", "convolutional network", "autoencoder", "transfer learning",
    "backpropagation", "hyperparameter tuning", "decision tree", "random forest", "XGBoost",
    "transformers", "natural language processing", "computer vision", "Bayesian networks",
    "GAN", "BERT", "GPT", "self-supervised learning", "semi-supervised learning"
]

# Compute Embeddings for ML Dictionary
ml_embeddings = model.encode(ml_terms, convert_to_tensor=True)

# Load a sample of the JSON dataset
def load_data(json_file, sample_size=100):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of research papers")

    return random.sample(data, min(sample_size, len(data)))  # Random sample of the dataset

# Compute relevance score using semantic similarity
def compute_relevance_score(title, user_input):
    title_embedding = model.encode(title, convert_to_tensor=True)
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    score = util.cos_sim(title_embedding, user_input_embedding).item()
    return round(score, 4)  # Round for readability

# Assign colors based on relevance score
def get_relevance_color(score):
    if score >= 0.5:
        return "green"
    elif 0.2 <= score < 0.5:
        return "yellow"
    else:
        return "red"

# Assign colors based on recency
def get_recent_color(timestamp, recent_threshold_days=3, old_threshold_days=90):
    now = datetime.utcnow()
    days_diff = (now - timestamp).days

    if days_diff <= recent_threshold_days:
        return "green"
    elif recent_threshold_days < days_diff <= old_threshold_days:
        return "yellow"
    else:
        return "red"

# Extract keyword trends using SBERT similarity
def extract_keyword_trends(data, user_input, top_n=20, similarity_threshold=0.5):  # Lowered threshold to capture more relevant data
    keyword_data = []

    for entry in data:
        title = entry.get('title', '')
        timestamp_str = entry.get('published', '')

        if not title or not timestamp_str:
            continue  

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            print(f"Skipping invalid timestamp: {timestamp_str}")
            continue

        # Compute relevance score with user input
        relevance_score = compute_relevance_score(title, user_input)

        # Compute title embedding and similarity with ML terms
        title_embedding = model.encode(title, convert_to_tensor=True)
        similarity_scores = util.cos_sim(title_embedding, ml_embeddings)
        max_sim_score = torch.max(similarity_scores).item()

        print(f"Title: {title}, Sim Score: {max_sim_score:.4f}, Relevance Score: {relevance_score:.4f}")

        if max_sim_score >= similarity_threshold and relevance_score >= 0.2:  # Only include those with sufficient relevance score
            keyword_data.append((title.lower(), timestamp, relevance_score, max_sim_score))

    if not keyword_data:
        print("No relevant keywords extracted! Debug: No keywords passed the threshold.")
        return []

    # Get top-N keywords
    keyword_counts = Counter([item[0] for item in keyword_data]).most_common(top_n)
    keyword_nodes = []

    for keyword, count in keyword_counts:
        keyword_entry = next(item for item in keyword_data if item[0] == keyword)
        timestamp = keyword_entry[1]
        relevance_score = keyword_entry[2]

        keyword_nodes.append({
            "id": keyword,
            "count": count,
            "timestamp": timestamp.isoformat(),
            "relevance_score": relevance_score,  # Newly computed relevance score
            "relevance_color": get_relevance_color(relevance_score),
            "recent_color": get_recent_color(timestamp),
            "similarity_score": keyword_entry[3]  # Store similarity score
        })

    return keyword_nodes

# Generate JSON for D3 Graph
def generate_graph_json(json_file, user_input, output_json="graph_data.json", sample_size=100):
    data = load_data(json_file, sample_size)
    keywords = extract_keyword_trends(data, user_input)

    if not keywords:
        print("No keywords found. The graph will contain only 'Research Topics'.")
        graph_data = {"nodes": [{"id": "Research Topics"}], "links": []}
    else:
        graph_data = {
            "nodes": [{"id": "Research Topics"}] + keywords,
            "links": [{"source": "Research Topics", "target": keyword["id"]} for keyword in keywords]
        }

    print(f"Total nodes: {len(graph_data['nodes'])}")
    print(f"Total links: {len(graph_data['links'])}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)

    print(f"Graph data saved to {output_json}")

# Example Usage
user_input = "I want to build a model that detects yoga poses using cnn and object detection"
generate_graph_json('arxiv_50k_papers.json', user_input, sample_size=1000)
