import cv2
import pytesseract
from PIL import Image

# Set path to Tesseract executable (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """Extract text from an image using OCR."""
    image = cv2.imread(image_path)  # Load image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray, timeout=5)  # Apply OCR
    return text

if __name__ == "__main__":
    img_path = r"C:\Users\91800\OneDrive\Desktop\NLP\data\docs-sm\invoice\0000333206.jpg"
    extracted_text = extract_text(img_path)
    print("Extracted Text:\n", extracted_text)
