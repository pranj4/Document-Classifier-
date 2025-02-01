from fastapi import FastAPI, UploadFile, File
from api.model_loader import classify_document
from api.invoice_extractor import extract_invoice_details

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Document Classification & Invoice Extraction API is Running!"}

@app.post("/classify_and_extract")
async def classify_and_extract(file: UploadFile = File(...)):  # expects file
    content = await file.read()
    text = content.decode("utf-8")  # Assuming the file is a text file

    # Classify the document
    category = classify_document(text)

    # If it's an invoice, extract the details
    if category == "invoice":
        extracted_data = extract_invoice_details(text)
        return {
            "category": category,
            "extracted_data": extracted_data
        }

    return {"category": category, "message": "No extraction needed"}
