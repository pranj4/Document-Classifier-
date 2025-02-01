import os
import pandas as pd
from src.extract_text import extract_text
from src.preprocess import clean_text

dataset_path = r"C:\Users\91800\OneDrive\Desktop\NLP\data\docs-sm"  # Path to dataset


def get_label_from_folder(folder_name):
    """Identify document type based on folder name."""
    if "invoice" in folder_name.lower():
        return "invoice"
    elif "budget" in folder_name.lower():
        return "budget"
    elif "email" in folder_name.lower():
        return "email"
    else:
        return "other"


# Process all images
data = []
for category in ["invoice", "budget", "email"]:  # 3 classes
    category_path = os.path.join(dataset_path, category)

    for img_file in os.listdir(category_path)[:125]:
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(category_path, img_file)

            extracted_text = extract_text(img_path)
            cleaned_text = clean_text(extracted_text)

            # Directly assign label based on folder name
            label = get_label_from_folder(category)  # Use category name as label

            data.append((cleaned_text, label))

# Save dataset
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("C:\\Users\\91800\\OneDrive\\Desktop\\NLP\\data\\dataset.csv", index=False)

print("✅ Dataset saved as dataset.csv")
