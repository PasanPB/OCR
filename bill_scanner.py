import pytesseract
from PIL import Image
import re
import joblib

# Load model & vectorizer
model = joblib.load("model/expense_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def scan_and_categorize(image_file):
    """Scan a bill image and return items with predicted categories."""
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)

    lines = text.split('\n')
    items = []
    for line in lines:
        match = re.match(r"(.+?)\s+(\d+\.\d{2})$", line.strip())
        if match:
            item = match.group(1).strip()
            price = float(match.group(2))
            category = model.predict(vectorizer.transform([item]))[0]
            items.append({
                "item": item,
                "price": price,
                "category": category
            })

    return {"items": items}

# Optional: run this file directly for testing
if __name__ == "__main__":
    print("🧠 Running bill_scanner.py standalone...\n")
    
    # OCR on sample bill image
    sample_image = "test_bills/sample_bill.jpg"
    image = Image.open(sample_image)
    text = pytesseract.image_to_string(image)

    # Print the raw extracted text (useful for debugging)
    print("📝 Extracted Text from Bill:\n")
    print(text)
    print("\n----------------------------\n")

    # Extract item names and prices
    lines = text.split('\n')
    items = []
    for line in lines:
        match = re.match(r"(.+?)\s+(\d+\.\d{2})$", line.strip())
        if match:
            item = match.group(1).strip()
            price = float(match.group(2))
            items.append((item, price))

    # If no items found
    if not items:
        print("⚠️ No items with prices found. Check OCR accuracy or bill format.")
    else:
        print("🧾 Detected Items and Predicted Categories:\n")
        for item, price in items:
            category = model.predict(vectorizer.transform([item]))[0]
            print(f"{item} - Rs.{price:.2f} → {category}")
