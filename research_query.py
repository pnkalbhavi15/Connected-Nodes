from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load GPT-2 (lightweight model for CPU usage)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")

def extract_research_topic(user_idea):
    """Generate research topic keywords using GPT-2."""
    prompt = f"Extract the main research topic from this research idea: {user_idea}\nTopic:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_length=50)
    extracted_topic = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Topic:")[-1].strip()
    
    return extracted_topic

# Example usage
if __name__ == "__main__":
    user_idea = input(" Describe your research idea: ")
    research_topic = extract_research_topic(user_idea)
    print(f"\n Extracted Research Topic: {research_topic}")
