import re

def clean_text(text):
    """Clean extracted text by removing unwanted characters."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text.strip()

if __name__ == "__main__":
    sample_text = "Invoice #12345 !! @Company Name"
    print("Cleaned Text:\n", clean_text(sample_text))
