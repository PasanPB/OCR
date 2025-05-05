import pytesseract
from PIL import Image
import re
import joblib

# Load model & vectorizer
model = joblib.load("model/expense_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# OCR on bill image
image = Image.open("test_bills/sample_bill.jpg")
text = pytesseract.image_to_string(image)

# Extract item names and prices
lines = text.split('\n')
items = []
for line in lines:
    match = re.match(r"(.+?)\s+(\d+\.\d{2})$", line.strip())
    if match:
        item = match.group(1).strip()
        price = float(match.group(2))
        items.append((item, price))

# Categorize
for item, price in items:
    category = model.predict(vectorizer.transform([item]))[0]
    print(f"{item} - Rs.{price:.2f} → {category}")
