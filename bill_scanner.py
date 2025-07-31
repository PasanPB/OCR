import pandas as pd
import numpy as np
import re
import json
import cv2
import pytesseract
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os

# Set Tesseract path (update if needed for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
# For Linux/Mac, comment out the above line or adjust as needed (e.g., /usr/bin/tesseract)

def preprocess_image(image_path):
    """Preprocess image for OCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Apply thresholding to improve text readability
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Resize to improve OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return img

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    try:
        img = preprocess_image(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        # Clean extracted text
        text = re.sub(r'\n\s*\n', '\n', text.strip())  # Remove extra newlines
        text = re.sub(r'[^\w\s.,]', '', text)  # Remove special characters
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def clean_text(text):
    """Clean text for ML processing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

def infer_expense_category(description):
    """Infer expense category based on item description."""
    if not isinstance(description, str):
        return "Miscellaneous"
    description = description.lower()
    if any(keyword in description for keyword in ["wine", "glass", "bottle", "rack", "opener", "kitchen", "cider", "dining table"]):
        return "Kitchenware"
    elif any(keyword in description for keyword in ["dress", "shoe", "cleat", "clothing", "outfit", "romper"]):
        return "Clothing"
    elif any(keyword in description for keyword in ["console", "computer", "xbox", "playstation", "nintendo", "desktop", "laptop"]):
        return "Electronics"
    elif any(keyword in description for keyword in ["book", "ebook", "hardcover", "paperback"]):
        return "Books"
    elif any(keyword in description for keyword in ["table", "sofa", "rug", "carpet", "furniture"]):
        return "Furniture"
    else:
        return "Miscellaneous"

# Load CSV data
csv_path = "batch_1.csv"  # Your CSV file
image_folder = "batch_1/"  # Your image folder
df = pd.read_csv(csv_path)

# Verify CSV columns
expected_columns = ['File Name', 'Json Data', 'OCRed Text']
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"CSV must contain columns: {expected_columns}")

# Extract item descriptions from Json Data
def extract_descriptions(json_str):
    try:
        json_data = json.loads(json_str.replace('""', '"'))
        items = json_data.get('items', [])
        descriptions = [item.get('description', '') for item in items]
        return ' '.join(descriptions)
    except json.JSONDecodeError:
        return ""

df['Item Descriptions'] = df['Json Data'].apply(extract_descriptions)

# Infer expense categories
df['Expense Type'] = df['Item Descriptions'].apply(infer_expense_category)

# Clean the OCRed text
df['Cleaned Text'] = df['OCRed Text'].apply(clean_text)

# Remove rows with empty text or invalid categories
df = df[df['Cleaned Text'].str.strip() != '']
df = df[df['Expense Type'] != '']
if df.empty:
    raise ValueError("No valid text data after cleaning. Check OCR output or CSV.")

# Prepare data for ML
X = df['Cleaned Text']
y = df['Expense Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Random Forest classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict expense type for a new bill image
def predict_expense_type(image_path):
    """Predict expense type for a new bill image."""
    text = extract_text_from_image(image_path)
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "Error: No text extracted"
    prediction = pipeline.predict([cleaned_text])[0]
    return prediction

# Example: Predict expense type for a new bill
new_bill_path = os.path.join(image_folder, "batch1-0494.jpg")  # Example image
if os.path.exists(new_bill_path):
    predicted_category = predict_expense_type(new_bill_path)
    print(f"Predicted Expense Type for {new_bill_path}: {predicted_category}")
else:
    print(f"New bill image not found: {new_bill_path}")

# Save the model
import joblib
joblib.dump(pipeline, "expense_categorizer_model.pkl")