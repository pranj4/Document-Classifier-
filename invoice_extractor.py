import re

def extract_invoice_details(text):
    # Improved regex for invoice number (allowing for multiple formats)
    invoice_number = re.search(r"(Invoice|Bill)[\s#]*[:\-]?\s*(\S+)", text, re.IGNORECASE)

    # Improved regex for date matching (more formats like DD/MM/YYYY, MM-DD-YYYY)
    date = re.search(r"\b(\d{2,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4}[-/.]\d{1,2})\b", text)

    # Improved regex for amounts (matches dollar amount, including optional spaces and commas)
    amount = re.search(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)

    # Improved vendor search to allow multiple words
    vendor = re.search(r"(Vendor|Supplier|Issued by|From)\s*[:\-]?\s*([\w\s]+)", text, re.IGNORECASE)

    return {
        "invoice_number": invoice_number.group(2) if invoice_number else None,
        "date": date.group(0) if date else None,
        "amount": amount.group(0) if amount else None,
        "vendor": vendor.group(2).strip() if vendor else None
    }

invoice_text = """
Invoice #12345
Date: 2025-01-15
Amount: $1,234.56
Vendor: ABC Supplies Ltd.
"""
result = extract_invoice_details(invoice_text)
print(result)
