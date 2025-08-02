import pandas as pd
import numpy as np
import re
import json
import cv2
import pytesseract
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import joblib
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Set Tesseract path (update if needed for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
# For Linux/Mac, comment out the above line or adjust as needed (e.g., /usr/bin/tesseract)

# MongoDB connection setup
MONGO_URI = "mongodb://localhost:27017"  # Update with your MongoDB URI (e.g., Atlas URI)
DB_NAME = "finance_app"
COLLECTION_NAME = "bills"

def connect_to_mongodb():
    """Connect to MongoDB and return the collection object."""
    print("Attempting to connect to MongoDB...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"Successfully connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        # Test insertion to verify write access
        test_doc = {"test": "connection_check", "timestamp": pd.Timestamp.now().isoformat()}
        collection.insert_one(test_doc)
        collection.delete_one({"test": "connection_check"})
        print(f"Test insertion and deletion successful in {DB_NAME}.{COLLECTION_NAME}")
        return collection
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        print("Ensure MongoDB server is running (mongod for local) or check MONGO_URI for Atlas.")
        print("For local MongoDB, run 'mongod' in a terminal.")
        print("For Atlas, verify the URI format: mongodb+srv://<username>:<password>@cluster0.mongodb.net")
        exit(1)
    except OperationFailure as e:
        print(f"Operation failed (possible authentication issue): {e}")
        print("Check MongoDB credentials or database permissions in Atlas.")
        exit(1)
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        exit(1)

def store_in_mongodb(collection, data):
    """Store data in MongoDB collection."""
    print(f"Attempting to store data for {data.get('File Name', 'unknown')} in MongoDB...")
    try:
        if isinstance(data, dict):  # Single document
            result = collection.insert_one(data)
            print(f"Stored document for {data['File Name']} in MongoDB (ID: {result.inserted_id})")
            # Verify insertion
            stored_doc = collection.find_one({"_id": result.inserted_id})
            if stored_doc:
                print(f"Verified: Document for {data['File Name']} exists in MongoDB")
            else:
                print(f"Warning: Document for {data['File Name']} not found after insertion")
        else:  # DataFrame
            records = data.to_dict('records')
            result = collection.insert_many(records)
            print(f"Stored {len(result.inserted_ids)} records in MongoDB collection: {DB_NAME}.{COLLECTION_NAME}")
    except OperationFailure as e:
        print(f"Operation failed while storing data in MongoDB: {e}")
        print("Check database permissions or MongoDB server status.")
    except Exception as e:
        print(f"Error storing data in MongoDB: {e}")

def preprocess_image(image_path):
    """Preprocess image for OCR with enhanced techniques."""
    print(f"Preprocessing image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
    img = cv2.fastNlMeansDenoising(img, h=15)  # Increased denoising strength
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return img

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    print(f"Extracting text from {image_path}...")
    try:
        img = preprocess_image(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        text = re.sub(r'\n\s*\n', '\n', text.strip())
        text = re.sub(r'[^\w\s.,]', '', text)
        print(f"Extracted Text from {image_path}: {text[:200]}...")
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def clean_text(text):
    """Clean text for ML processing."""
    print("Cleaning extracted text...")
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    cleaned_text = text.strip()
    print(f"Cleaned Text: {cleaned_text[:200]}...")
    return cleaned_text

def infer_expense_category(description):
    """Infer expense category based on item description with expanded keywords."""
    print("Inferring expense category...")
    if not isinstance(description, str):
        return "Miscellaneous"
    description = description.lower()
    if any(keyword in description for keyword in ["food", "grocery", "bread", "milk", "cheese", "meat", "vegetable", "fruit", "cereal", "snack", "beverage", "coffee", "tea", "chicken", "rice", "pizza", "burger", "sushi", "pasta"]):
        return "Food"
    elif any(keyword in description for keyword in ["wine", "glass", "bottle", "rack", "opener", "kitchen", "cider", "dining table", "plate", "cutlery", "pan", "pot", "knife", "spatula", "mixing bowl", "toaster", "blender"]):
        return "Kitchenware"
    elif any(keyword in description for keyword in ["dress", "shoe", "cleat", "clothing", "outfit", "romper", "shirt", "pants", "jacket", "socks", "sweater", "t-shirt", "jeans", "hat", "scarf", "belt", "blouse", "skirt", "trousers", "coat", "sweatshirt", "hoodie", "leggings", "jumper", "sneakers"]):
        return "Clothing"
    elif any(keyword in description for keyword in ["console", "computer", "xbox", "playstation", "nintendo", "desktop", "laptop", "phone", "tablet", "tv", "headphones", "speaker", "smartwatch", "camera", "mouse", "keyboard", "monitor"]):
        return "Electronics"
    elif any(keyword in description for keyword in ["book", "ebook", "hardcover", "paperback", "novel", "textbook", "comic", "magazine", "cookbook", "reference", "study guide", "poetry", "biography", "science", "history", "fiction"]):
        return "Books"
    elif any(keyword in description for keyword in ["table", "sofa", "rug", "carpet", "furniture", "chair", "bed", "desk", "bookshelf", "wardrobe", "nightstand", "lamp", "stool", "cabinet", "mattress"]):
        return "Furniture"
    else:
        return "Miscellaneous"

# Load CSV data for training
csv_path = "batch_1.csv"
image_folder = "batch_1/"
print(f"Loading training data from {csv_path}...")
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
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
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

# Connect to MongoDB
collection = connect_to_mongodb()

# Prepare data for ML
X = df['Cleaned Text']
y = df['Expense Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Random Forest classifier
print("Training machine learning model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words=['english', 'total', 'store', 'receipt', 'tax', 'cash', 'payment'])),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to get top TF-IDF features
def get_top_features(pipeline, text, n=5):
    tfidf = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    text_vector = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    feature_importance = text_vector.toarray()[0]
    top_indices = feature_importance.argsort()[-n:][::-1]
    return [(feature_names[i], feature_importance[i]) for i in top_indices]

# Function to predict expense type for a new bill image
def predict_expense_type(image_path, collection):
    """Predict expense type for a new bill image and store in MongoDB."""
    print(f"Processing new bill: {image_path}")
    try:
        text = extract_text_from_image(image_path)
        cleaned_text = clean_text(text)
        if not cleaned_text:
            print("Error: No text extracted from image")
            # Store error document for debugging
            bill_data = {
                "File Name": os.path.basename(image_path),
                "Json Data": "",
                "OCRed Text": text,
                "Item Descriptions": "",
                "Expense Type": "Error",
                "Prediction Probabilities": {},
                "Top TF-IDF Features": {},
                "Timestamp": pd.Timestamp.now().isoformat(),
                "Error": "No text extracted from image"
            }
            store_in_mongodb(collection, bill_data)
            return None
        # Get prediction and probabilities
        prediction = pipeline.predict([cleaned_text])[0]
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        class_labels = pipeline.classes_
        print("Prediction Probabilities:")
        for label, prob in zip(class_labels, probabilities):
            print(f"{label}: {prob:.2f}")
        # Log top TF-IDF features
        print("Top TF-IDF Features:")
        top_features = get_top_features(pipeline, cleaned_text)
        for feature, score in top_features:
            print(f"{feature}: {score:.4f}")
        # Check confidence threshold
        max_prob = max(probabilities)
        if max_prob < 0.3:
            print(f"Warning: Low confidence prediction ({max_prob:.2f}). Consider adding more training data.")
        # Store in MongoDB
        bill_data = {
            "File Name": os.path.basename(image_path),
            "Json Data": "",  # Update if you have JSON data for the new bill
            "OCRed Text": text,
            "Item Descriptions": cleaned_text,
            "Expense Type": prediction,
            "Prediction Probabilities": {label: float(prob) for label, prob in zip(class_labels, probabilities)},
            "Top TF-IDF Features": {feature: float(score) for feature, score in top_features},
            "Timestamp": pd.Timestamp.now().isoformat()
        }
        store_in_mongodb(collection, bill_data)
        return prediction
    except Exception as e:
        print(f"Error processing bill {image_path}: {e}")
        # Store error document for debugging
        bill_data = {
            "File Name": os.path.basename(image_path),
            "Json Data": "",
            "OCRed Text": "",
            "Item Descriptions": "",
            "Expense Type": "Error",
            "Prediction Probabilities": {},
            "Top TF-IDF Features": {},
            "Timestamp": pd.Timestamp.now().isoformat(),
            "Error": str(e)
        }
        store_in_mongodb(collection, bill_data)
        return None

# Loop to scan multiple bills
image_folder = "batch_1/"
while True:
    print(f"\nEnter the name of the new bill image in the '{image_folder}' folder (e.g., clothing_bill.jpg) or 'quit' to exit:")
    new_bill_name = input().strip()
    if new_bill_name.lower() == 'quit':
        break
    new_bill_path = os.path.join(image_folder, new_bill_name)
    print(f"Checking path: {os.path.abspath(new_bill_path)}")
    if os.path.exists(new_bill_path):
        predicted_category = predict_expense_type(new_bill_path, collection)
        if predicted_category:
            print(f"Predicted Expense Type for {new_bill_path}: {predicted_category}")
    else:
        print(f"New bill image not found: {new_bill_path}")
        print(f"Please place the image in the '{image_folder}' folder and try again.")

# Save the model
joblib.dump(pipeline, "expense_categorizer_model.pkl")

# Generate expense_categories.csv
categories_data = {
    "Category": ["Food", "Kitchenware", "Clothing", "Electronics", "Books", "Furniture", "Miscellaneous"],
    "Keywords": [
        "food,grocery,bread,milk,cheese,meat,vegetable,fruit,cereal,snack,beverage,coffee,tea,chicken,rice,pizza,burger,sushi,pasta",
        "wine,glass,bottle,rack,opener,kitchen,cider,dining table,plate,cutlery,pan,pot,knife,spatula,mixing bowl,toaster,blender",
        "dress,shoe,cleat,clothing,outfit,romper,shirt,pants,jacket,socks,sweater,t-shirt,jeans,hat,scarf,belt,blouse,skirt,trousers,coat,sweatshirt,hoodie,leggings,jumper,sneakers",
        "console,computer,xbox,playstation,nintendo,desktop,laptop,phone,tablet,tv,headphones,speaker,smartwatch,camera,mouse,keyboard,monitor",
        "book,ebook,hardcover,paperback,novel,textbook,comic,magazine,cookbook,reference,study guide,poetry,biography,science,history,fiction",
        "table,sofa,rug,carpet,furniture,chair,bed,desk,bookshelf,wardrobe,nightstand,lamp,stool,cabinet,mattress",
        "none (default for unmatched descriptions)"
    ]
}
categories_df = pd.DataFrame(categories_data)
categories_df.to_csv("expense_categories.csv", index=False)
print("Generated expense_categories.csv with all categories and keywords.")