import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

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
joblib.dump(model, 'expense_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
