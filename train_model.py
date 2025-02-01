import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
df = pd.read_csv(r"C:\Users\91800\OneDrive\Desktop\NLP\data\dataset.csv")

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

test_texts = test_texts.fillna('')  # Replace NaN with empty strings
train_texts = train_texts.fillna('')

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Check for any non-string entries in test_texts
non_string_entries = [text for text in test_texts if not isinstance(text, str)]
if non_string_entries:
    print(f"Non-string entries found in test_texts: {non_string_entries}")

# Ensure test_texts and train_texts are in the correct format
assert isinstance(train_texts, pd.Series)  # Ensure train_texts is a pandas Series
assert isinstance(test_texts, pd.Series)  # Ensure test_texts is a pandas Series
assert all(isinstance(text, str) for text in train_texts)  # Check if all train texts are strings

# Tokenize text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

label_map = {"invoice": 0, "budget": 1, "email": 2}
train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]

# Custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Prepare dataset for training and testing
train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="../model/results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
      # 🔥 Reduce learning rate (Default is 5e-5, but 2e-5 works better for fine-tuning)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


trainer.train()

# Save model and tokenizer
model.save_pretrained(r"./saved_model")
tokenizer.save_pretrained(r"./saved_model")

# Evaluate the model
trainer.evaluate()

# Predict on the test set
model.eval()
predictions = []
true_labels = []

for batch in test_dataset:
    input_ids = batch['input_ids'].unsqueeze(0)
    attention_mask = batch['attention_mask'].unsqueeze(0)
    labels = batch['labels']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class_ids = torch.argmax(logits, dim=-1)

    predictions.append(predicted_class_ids.item())
    true_labels.append(labels.item())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)

# Calculate precision, recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

# Print the evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

