import json
import torch
import random
import pandas as pd
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from dateutil import parser

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
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of research papers")

        return random.sample(data, min(sample_size, len(data)))  # Random sample of the dataset
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Error loading JSON file: {e}")
        return []

# Compute relevance score using semantic similarity
def compute_relevance_score(title, user_input):
    title_embedding = model.encode(title, convert_to_tensor=True)
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    score = util.cos_sim(title_embedding, user_input_embedding).item()
    return round(score, 4)

# Assign colors based on relevance score
def get_relevance_color(score):
    if score >= 0.7:
        return "green"
    elif 0.4 <= score < 0.7:
        return "yellow"
    else:
        return "red"

# Assign colors based on recency
def get_recent_color(timestamp, recent_threshold_days=30, old_threshold_days=90):
    now = datetime.utcnow()
    days_diff = (now - timestamp).days

    if days_diff <= recent_threshold_days:
        return "green"
    elif recent_threshold_days < days_diff <= old_threshold_days:
        return "yellow"
    else:
        return "red"

# Extract keyword trends using SBERT similarity
def extract_keyword_trends(data, user_input, top_n=20, similarity_threshold=0.5):
    keyword_data = []

    for entry in data:
        title = entry.get('title', '')
        timestamp_str = entry.get('published', '')

        if not title or not timestamp_str:
            continue  

        try:
            timestamp = parser.parse(timestamp_str)  # Flexible timestamp parsing
        except ValueError:
            print(f"Skipping invalid timestamp: {timestamp_str}")
            continue

        # Compute relevance score
        relevance_score = compute_relevance_score(title, user_input)

        # Compute similarity with ML terms
        title_embedding = model.encode(title, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(title_embedding, ml_embeddings).squeeze()
        max_sim_score = similarity_scores.max().item()

        print(f"Title: {title}, Sim Score: {max_sim_score:.4f}, Relevance Score: {relevance_score:.4f}")

        if max_sim_score >= similarity_threshold:
            keyword_data.append((title.lower(), timestamp, relevance_score, max_sim_score))

    if not keyword_data:
        print("No relevant keywords extracted! Adding a placeholder node.")
        return [{
            "id": "No Relevant Keywords Found",
            "count": 1,
            "timestamp": datetime.utcnow().isoformat(),
            "relevance_score": 0,
            "relevance_color": "red",
            "recent_color": "red",
            "similarity_score": 0
        }]

    keyword_counts = Counter([item[0] for item in keyword_data]).most_common(top_n)
    return [
        {
            "id": keyword,
            "count": count,
            "timestamp": timestamp.isoformat(),
            "relevance_score": relevance_score,
            "relevance_color": get_relevance_color(relevance_score),
            "recent_color": get_recent_color(timestamp),
            "similarity_score": similarity_score
        }
        for keyword, count in keyword_counts
        for timestamp, relevance_score, similarity_score in [next(item[1:] for item in keyword_data if item[0] == keyword)]
    ]

# Generate JSON for D3 Graph
def generate_graph_json(json_file, user_input, output_json="graph_data.json", sample_size=100):
    data = load_data(json_file, sample_size)
    keywords = extract_keyword_trends(data, user_input)

    graph_data = {
        "nodes": [{"id": "Research Topics", "group": 1}] + [{"id": k["id"], "group": 2} for k in keywords],
        "links": [{"source": "Research Topics", "target": k["id"]} for k in keywords]
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)

    print(f"✅ Graph data saved to {output_json}")

# 🟢 Take user input from the terminal
if __name__ == "__main__":
    user_input = input("\n🔍 Enter your research query: ").strip()
    
    if not user_input:
        print("❌ Query cannot be empty. Please enter a valid research topic.")
    else:
        generate_graph_json('data/datasets1.json', user_input, sample_size=1000)
