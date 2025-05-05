import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data with potential class imbalance
data = {
    'item': ['Ginger mule', 'tonic', 'vodka', 'Uber', 'Taxi', 'Electric Bill', 'Water Bill', 'Movie'],
    'category': ['Food', 'Food', 'Food', 'Transport', 'Transport', 'Utilities', 'Utilities', 'Entertainment']
}
df = pd.DataFrame(data)

# 1. Check class distribution
print("Original class distribution:")
print(df['category'].value_counts())

# 2. Data augmentation for small classes
def augment_data(df, min_samples=3):
    augmented = []
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        n_samples = len(category_df)
        
        if n_samples < min_samples:
            # Duplicate existing samples with small variations
            n_needed = min_samples - n_samples
            for _ in range(n_needed):
                sample = category_df.sample(1)
                # Create variation by adding random word
                new_item = sample['item'].values[0] + " " + np.random.choice(['payment', 'service', 'purchase'])
                augmented.append({'item': new_item, 'category': category})
    
    if augmented:
        return pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)
    return df

df = augment_data(df, min_samples=2)
print("\nAugmented class distribution:")
print(df['category'].value_counts())

# 3. Proceed only if all classes have ≥2 samples
if (df['category'].value_counts() >= 2).all():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['item'])
    y = df['category']
    
    # Use stratified split only if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("\nWarning: Couldn't stratify - some classes still too small")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("\nModel trained successfully!")
else:
    print("\nError: Some categories still have fewer than 2 samples")
    print("Please add more data for these categories:")
    print(df['category'].value_counts()[df['category'].value_counts() < 2])