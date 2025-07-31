import pytesseract
from PIL import Image, ImageEnhance
import re
import joblib
import numpy as np

class RestaurantBillScanner:
    def __init__(self):
        self.model = joblib.load('restaurant_expense_classifier.pkl')
        self.vectorizer = joblib.load('restaurant_vectorizer.pkl')
        self.drink_keywords = {
            'gin', 'tonic', 'mule', 'vodka', 'whiskey', 'rum', 
            'wine', 'beer', 'martini', 'margarita', 'mojito'
        }

    def enhance_image(self, image_path):
        """Specialized enhancement for receipt paper"""
        img = Image.open(image_path)
        img = img.convert('L')  # Grayscale
        
        # Boost low-contrast receipt text
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(3.0)
        
        # Sharpen faded text
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(2.5)

    def extract_line_items(self, text):
        """Specialized parser for restaurant bills"""
        items = []
        current_item = ""
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Price detection (supports $10.50 and 10.50 formats)
            price_match = re.search(r'(\$?\s*\d+\.\d{2})\b', line)
            
            if price_match:
                price = float(price_match.group(1).replace('$', ''))
                item = current_item + " " + line[:price_match.start()].strip()
                items.append({'item': item, 'price': price})
                current_item = ""
            else:
                # Handle multi-line items (like "Hendrick Gin\n& Tonic")
                if line and not any(line.lower().startswith(x) for x in ['sub', 'total', 'tax', 'balance']):
                    current_item += " " + line if current_item else line
        
        return items

    def classify_item(self, text):
        """Enhanced classification for drinks"""
        # Preprocess for alcohol detection
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Check for obvious drink keywords first
        if any(keyword in cleaned for keyword in self.drink_keywords):
            return 'Alcohol', 0.95  # High confidence
            
        # Use ML model for others
        vec = self.vectorizer.transform([cleaned])
        return self.model.predict(vec)[0], self.model.predict_proba(vec).max()

    def process_bill(self, image_path):
        """Complete processing pipeline"""
        try:
            # OCR with receipt-optimized settings
            img = self.enhance_image(image_path)
            text = pytesseract.image_to_string(img, config='--psm 4 --oem 3')
            
            # Extract items with prices
            line_items = self.extract_line_items(text)
            
            # Classify each item
            results = []
            for item in line_items:
                category, confidence = self.classify_item(item['item'])
                results.append({
                    'description': item['item'].strip(),
                    'amount': item['price'],
                    'category': category,
                    'confidence': float(confidence)
                })
            
            return {
                'status': 'success',
                'items': results,
                'raw_text': text
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

# Example Usage
if __name__ == "__main__":
    scanner = RestaurantBillScanner()
    result = scanner.process_bill("test_bills/sample_bill.jpg")
    
    if result['status'] == 'success':
        print("BILL ANALYSIS RESULTS:")
        print("-" * 40)
        for item in result['items']:
            print(f"{item['description']: <30} ${item['amount']: >6.2f} → {item['category']} ({item['confidence']:.0%})")
        print("\nRAW TEXT EXTRACTED:")
        print("-" * 40)
        print(result['raw_text'])
    else:
        print("Error:", result['message'])