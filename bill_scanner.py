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
import joblib

# Set Tesseract path (update if needed for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
# For Linux/Mac, comment out the above line or adjust as needed (e.g., /usr/bin/tesseract)

def preprocess_image(image_path):
    """Preprocess image for OCR with enhanced techniques."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Denoise to reduce noise
    img = cv2.fastNlMeansDenoising(img, h=10)
    # Adaptive thresholding for better contrast
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
        print(f"Extracted Text from {image_path}: {text[:200]}...")  # Debug: Print first 200 chars
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
    """Infer expense category based on item description with expanded keywords."""
    if not isinstance(description, str):
        return "Miscellaneous"
    description = description.lower()
    if any(keyword in description for keyword in ["food", "grocery", "bread", "milk", "cheese", "meat", "vegetable", "fruit", "cereal", "snack", "beverage", "coffee", "tea", "chicken", "rice", "pizza", "burger"]):
        return "Food"
    elif any(keyword in description for keyword in ["wine", "glass", "bottle", "rack", "opener", "kitchen", "cider", "dining table", "plate", "cutlery", "pan", "pot"]):
        return "Kitchenware"
    elif any(keyword in description for keyword in ["dress", "shoe", "cleat", "clothing", "outfit", "romper", "shirt", "pants", "jacket", "socks"]):
        return "Clothing"
    elif any(keyword in description for keyword in ["console", "computer", "xbox", "playstation", "nintendo", "desktop", "laptop", "phone", "tablet", "tv"]):
        return "Electronics"
    elif any(keyword in description for keyword in ["book", "ebook", "hardcover", "paperback", "novel", "textbook"]):
        return "Books"
    elif any(keyword in description for keyword in ["table", "sofa", "rug", "carpet", "furniture", "chair", "bed", "desk"]):
        return "Furniture"
    else:
        return "Miscellaneous"

# Load CSV data
csv_path = "batch_1.csv"  # Your CSV file
image_folder = "batch_1/"  # Your image folder
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Ensure the file is in the project directory.")
    exit(1)

# Verify CSV columns
expected_columns = ['File Name', 'Json Data', 'OCRed Text']
if not all(col in df.columns for col in expected_columns):
    print(f"Error: CSV must contain columns: {expected_columns}")
    print(f"Actual columns found: {list(df.columns)}")
    print("Please check column names in batch_1.csv and ensure they match exactly.")
    exit(1)

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

# Check category distribution
print("Category Distribution in Training Data:")
print(df['Expense Type'].value_counts())

# Clean the item descriptions for training
df['Cleaned Text'] = df['Item Descriptions'].apply(clean_text)

# Remove rows with empty text or invalid categories
df = df[df['Cleaned Text'].str.strip() != '']
df = df[df['Expense Type'] != '']
if df.empty:
    print("Error: No valid text data after cleaning. Check JSON data or CSV.")
    exit(1)

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
def predict_expense_type(image_path, manual_text=None):
    """Predict expense type for a new bill image or manual text."""
    if manual_text:
        text = manual_text
        print(f"Using manual text: {text[:200]}...")
    else:
        text = extract_text_from_image(image_path)
    cleaned_text = clean_text(text)
    print(f"Cleaned Text: {cleaned_text[:200]}...")  # Debug: Print cleaned text
    if not cleaned_text:
        return "Error: No text extracted"
    # Get prediction and probabilities
    prediction = pipeline.predict([cleaned_text])[0]
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    class_labels = pipeline.classes_
    print("Prediction Probabilities:")
    for label, prob in zip(class_labels, probabilities):
        print(f"{label}: {prob:.2f}")
    # Check confidence threshold
    max_prob = max(probabilities)
    if max_prob < 0.3:
        print(f"Warning: Low confidence prediction ({max_prob:.2f}). Consider adding more training data.")
    return prediction

# Example: Predict expense type for a new bill
new_bill_path = os.path.join(image_folder, "restaurantsixth.jpg")  # Update with your image name
print(f"Checking path: {os.path.abspath(new_bill_path)}")  # Debug: Print absolute path
if os.path.exists(new_bill_path):
    predicted_category = predict_expense_type(new_bill_path)
    print(f"Predicted Expense Type for {new_bill_path}: {predicted_category}")
else:
    print(f"New bill image not found: {new_bill_path}")

# Test with manual text to verify "food" categorization
manual_text = "Food, Bread, Milk, Chicken"  # Example food-related text
predicted_category = predict_expense_type(new_bill_path, manual_text=manual_text)
print(f"Predicted Expense Type with manual text: {predicted_category}")

# Save the model
joblib.dump(pipeline, "expense_categorizer_model.pkl")

# Generate expense_categories.csv
categories_data = {
    "Category": ["Food", "Kitchenware", "Clothing", "Electronics", "Books", "Furniture", "Miscellaneous"],
    "Keywords": [
        "food,grocery,bread,milk,cheese,meat,vegetable,fruit,cereal,snack,beverage,coffee,tea,chicken,rice,pizza,burger",
        "wine,glass,bottle,rack,opener,kitchen,cider,dining table,plate,cutlery,pan,pot",
        "dress,shoe,cleat,clothing,outfit,romper,shirt,pants,jacket,socks",
        "console,computer,xbox,playstation,nintendo,desktop,laptop,phone,tablet,tv",
        "book,ebook,hardcover,paperback,novel,textbook",
        "table,sofa,rug,carpet,furniture,chair,bed,desk",
        "none (default for unmatched descriptions)"
    ]
}
categories_df = pd.DataFrame(categories_data)
categories_df.to_csv("expense_categories.csv", index=False)
print("Generated expense_categories.csv with all categories and keywords.")