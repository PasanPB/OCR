import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Sample training data
data = {
    'item': ['Pizza', 'Uber Ride', 'Electricity Bill', 'Movie Ticket', 'Shampoo', 'Milk'],
    'category': ['Food', 'Transport', 'Utilities', 'Entertainment', 'Personal Care', 'Grocery']
}
df = pd.DataFrame(data)

# Convert item names to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['item'])
y = df['category']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and vectorizer
output_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(output_dir, 'expense_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))

print("✅ Model and vectorizer saved successfully!")
