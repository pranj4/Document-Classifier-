import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load trained model & tokenizer
model_path = r"C:\Users\91800\OneDrive\Desktop\NLP\src\saved_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

label_map = {0: "invoice", 1: "budget", 2: "email"}


def classify_document(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move inputs to the same device as the model (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():  # No need to compute gradients during inference
        outputs = model(**inputs)

    # Get predicted class label
    prediction = outputs.logits.argmax(dim=-1).item()

    # Map prediction to category
    return label_map[prediction]
