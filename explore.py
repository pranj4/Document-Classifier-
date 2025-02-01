import os

DATA_DIR = "C:\\Users\\91800\\OneDrive\\Desktop\\NLP\\data\\docs-sm"

# List categories
categories = os.listdir(DATA_DIR)
print("📂 Available Categories:", categories)

# List sample files from each category
for category in categories:
    category_path = os.path.join(DATA_DIR, category)
    if os.path.isdir(category_path):
        sample_files = os.listdir(category_path)[:5]  # Show first 5 files
        print(f"\n📄 {category} Samples: {sample_files}")
