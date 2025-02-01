import pandas as pd
import re

def extract_invoice_details(text):
    invoice_number = re.search(r"(?i)(invoice|inv|bill)\s*#?\s*[:\-]?\s*(\w+)", text)
    invoice_date = re.search(r"(?i)(date|invoice date)\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", text)
    amount = re.search(r"(?i)(total|amount due|grand total)\s*[:\-]?\s*\$?([\d,]+\.\d{2})", text)
    vendor = re.search(r"(?i)(vendor|supplier|billed to|from)\s*[:\-]?\s*([\w\s,&]+)", text)

    return {
        "invoice_number": invoice_number.group(2) if invoice_number else None,
        "invoice_date": invoice_date.group(2) if invoice_date else None,
        "amount": amount.group(2) if amount else None,
        "vendor": vendor.group(2) if vendor else None
    }

# Test on sample extracted text
sample_text = """
Invoice # 12345
Invoice Date: 10/12/2024
Amount Due: $1,234.56
Vendor: ABC Corp
"""

print(extract_invoice_details(sample_text))



# Load dataset
df = pd.read_csv(r"C:\Users\91800\OneDrive\Desktop\NLP\data\dataset.csv")

# Create a copy to avoid warnings
df_invoice = df[df["label"] == "invoice"].copy()

# Apply extraction
df_invoice["extracted"] = df_invoice["text"].fillna("").apply(extract_invoice_details)

# Expand extracted data into separate columns
df_invoice_extracted = df_invoice.join(pd.DataFrame(df_invoice.pop("extracted").tolist()))

# Save results
df_invoice_extracted.to_csv(r"C:\Users\91800\OneDrive\Desktop\NLP\src\results\extracted_invoices.csv", index=False)

print("Extraction completed. Check results/extracted_invoices.csv")