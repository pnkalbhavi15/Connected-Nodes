import json
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter

# Load dataset
file_path = "C:\\Users\\Hp\\Desktop\\SEM 6\\confluence\\Connected-Nodes\\arxiv_50k_papers.json"  # Update if needed
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract titles & years
papers = [(paper["title"], paper["published"][:4]) for paper in data]  # Extract title & year

# Define AI/ML concepts mapping
concepts_map = {
    "diffusion": "Diffusion Models",
    "language model": "Large Language Models (LLMs)",
    "GAN": "Generative Adversarial Networks (GANs)",
    "attention": "Transformers & Attention Mechanisms",
    "variational": "Variational Models",
    "tracking": "Neural Tracking",
    "generative": "Generative Models",
    "zero-shot": "Zero-Shot Learning",
    "flow-based": "Flow-Based Models",
    "rectified flow": "Flow Matching",
    "manipulation": "Robotics & Dexterous Manipulation",
    "conditional prior": "Conditional Generative Models"
}

# Function to extract concepts from titles
def extract_concepts(title):
    detected_concepts = set()
    for keyword, concept in concepts_map.items():
        if re.search(rf"\b{keyword}\b", title, re.IGNORECASE):
            detected_concepts.add(concept)
    return detected_concepts

# Count occurrences of each concept per year
yearly_trends = defaultdict(Counter)
for title, year in papers:
    detected_concepts = extract_concepts(title)
    yearly_trends[year].update(detected_concepts)

# Convert to Pandas DataFrame for analysis
df_trends = pd.DataFrame(yearly_trends).fillna(0).T.sort_index()

# Plot trends over time
plt.figure(figsize=(12, 6))
for concept in concepts_map.values():
    if concept in df_trends.columns:
        plt.plot(df_trends.index, df_trends[concept], label=concept, marker="o")

plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.title("Trending AI/ML Concepts Over Time (Based on ArXiv Data)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Print yearly trend summary
print("📌 Yearly AI/ML Trends:")
trend_data = {}  # Prepare data for JSON storage
for year in sorted(yearly_trends.keys(), reverse=True):
    print(f"\n🔹 {year}:")
    top_concepts = {concept: count for concept, count in yearly_trends[year].most_common(5)}  # Top 5 per year
    trend_data[year] = top_concepts
    for concept, count in top_concepts.items():
        print(f"   - {concept}: {count} papers")

# Save to a JSON file
output_file = "C:\\Users\\Hp\\Desktop\\SEM 6\\confluence\\Connected-Nodes\\output_trend.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(trend_data, f, indent=4, ensure_ascii=False)

print(f"✅ Output saved to {output_file}")
