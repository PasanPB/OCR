import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

# Specialized training data for restaurant/bar receipts
data = {
    'item': [
        # Alcohol
        'Hendrick Gin Tonic', 'Ginger Mule', 'Glass Camus Zin', 'Titos Vodka Soda',
        'Jack Daniels', 'Rum Coke', 'Vodka Martini', 'Whiskey Sour', 'Chardonnay Glass',
        'Craft Beer', 'House Wine', 'Margarita', 'Mojito', 'Old Fashioned',
        
        # Food
        'Steak Dinner', 'Caesar Salad', 'Cheeseburger', 'Chicken Wings',
        'Pasta Carbonara', 'Mushroom Risotto', 'Soup of the Day', 'Dessert Platter',
        
        # Venue Specific
        'Room Service', 'Minibar Charge', 'Banquet Fee', 'Service Charge'
    ],
    'category': [
        'Alcohol', 'Alcohol', 'Alcohol', 'Alcohol',
        'Alcohol', 'Alcohol', 'Alcohol', 'Alcohol', 'Alcohol',
        'Alcohol', 'Alcohol', 'Alcohol', 'Alcohol', 'Alcohol',
        
        'Food', 'Food', 'Food', 'Food',
        'Food', 'Food', 'Food', 'Food',
        
        'Hotel', 'Hotel', 'Hotel', 'Hotel'
    ]
}

df = pd.DataFrame(data)

# Enhanced preprocessing for drink names
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\b(glass|bottle|shot|double)\b', '', text)  # Remove quantity words
    return text.strip()

df['item_clean'] = df['item'].apply(preprocess_text)

# Specialized vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),  # Capture multi-word drink names
    stop_words=['and', 'the', 'with']
)
X = vectorizer.fit_transform(df['item_clean'])
y = df['category']

# Model optimized for drink classification
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight='balanced'
)
model.fit(X, y)

# Save model
joblib.dump(model, 'restaurant_expense_classifier.pkl')
joblib.dump(vectorizer, 'restaurant_vectorizer.pkl')

print("Model trained with specialized drink recognition!")