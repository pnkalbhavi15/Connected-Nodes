import torch
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "models/query_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_training_data():
    data = [
        {"query": "Recent advancements in AI", "label": 0},
        {"query": "Machine learning applications in healthcare", "label": 1},
        {"query": "What is reinforcement learning?", "label": 2},
        {"query": "Latest papers on quantum computing", "label": 3},
        {"query": "Explain neural networks", "label": 4},
    ]
    return data

# Load dataset
data = load_training_data()
queries = [d["query"] for d in data]
labels = [d["label"] for d in data]

train_texts, val_texts, train_labels, val_labels = train_test_split(queries, labels, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_labels})
val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "attention_mask": val_encodings["attention_mask"], "labels": val_labels})

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Model training complete! Saved at {OUTPUT_DIR}")
