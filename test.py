'''import os


dataset_path = "/config\\data\\docs-sm"

files = os.listdir(dataset_path)
print(f"Total documents: {len(files)}")
print("Sample files:", files[:5])


df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("dataset.csv", index=False)

print("✅ Dataset saved as dataset.csv")
print(df.head())  # Print first 5 rows for verification'''

import pandas as pd

df = pd.read_csv(r"C:\Users\91800\OneDrive\Desktop\NLP\src\results\extracted_invoices.csv")  # Update the path if needed
print(df)

import csv

with open(r"C:\Users\91800\OneDrive\Desktop\NLP\src\results\extracted_invoices.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)