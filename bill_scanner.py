import pytesseract
from PIL import Image, ImageEnhance
import re
import joblib
import numpy as np

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_tesseract_executable>'

def preprocess_image(image_path):
    """Enhance image for better OCR results"""
    image = Image.open(image_path)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    return image

def extract_items(text):
    """Improved item extraction with multiple patterns"""
    items = []
    patterns = [
        r"^([^\d$]+?)\s+([$\-]?\s*\d+\.\d{2})\b",  # Standard item with price
        r"^([^\d$]+?)\s+[xX*]\s*\d+\s+([$\-]?\s*\d+\.\d{2})",  # With quantity
        r"^([^\d$]+?)\s+-\s+([$\-]?\s*\d+\.\d{2})",  # Dash separator
        r"^([^\d$]+?)\s+@\s+\d+\.\d{2}\s+([$\-]?\s*\d+\.\d{2})",  # Unit pricing
        r"^([^\d$]+?)\s+(\d+)\s+([$\-]?\s*\d+\.\d{2})"  # Item with count
    ]
    
    for line in text.split('\n'):
        line = line.strip()
        if not line or any(skip in line.lower() for skip in ['subtotal', 'total', 'tax', 'tip', 'change']):
            continue
            
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    item = match.group(1).strip()
                    price = float(match.group(-1).replace('$', '').replace(' ', ''))
                    items.append({'item': item, 'price': price, 'raw': line})
                    break
                except (ValueError, IndexError):
                    continue
    return items

def scan_receipt(image_path):
    """Complete receipt scanning pipeline"""
    try:
        # Load ML model
        model = joblib.load("model/expense_classifier.pkl")
        vectorizer = joblib.load("model/vectorizer.pkl")
        
        # Preprocess and OCR
        image = preprocess_image(image_path)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$.- '
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Extract items
        items = extract_items(text)
        
        if not items:
            # Try alternative OCR approach if no items found
            alt_config = r'--oem 3 --psm 11'
            text = pytesseract.image_to_string(image, config=alt_config)
            items = extract_items(text)
            
        # Classify items
        results = []
        for item in items:
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', item['item'].lower())
            vec = vectorizer.transform([cleaned])
            category = model.predict(vec)[0]
            proba = model.predict_proba(vec).max()
            results.append({
                'item': item['item'],
                'price': item['price'],
                'category': category,
                'confidence': f"{proba:.1%}",
                'raw_text': item['raw']
            })
        
        return {
            'status': 'success',
            'items': results,
            'raw_text': text
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'items': [],
            'raw_text': ''
        }

# Example usage
if __name__ == "__main__":
    result = scan_receipt("test_bills/sample_bill.jpg")
    
    if result['status'] == 'success':
        print("✅ Receipt processed successfully!")
        print("\n📝 Raw OCR Text:\n")
        print(result['raw_text'])
        
        if result['items']:
            print("\n🧾 Detected Items:")
            for item in result['items']:
                print(f"- {item['item']}: ${item['price']:.2f} ({item['category']}, {item['confidence']} confidence)")
                print(f"  Raw: {item['raw_text']}")
        else:
            print("\n⚠️ No items found. Possible issues:")
            print("1. Poor image quality")
            print("2. Unsupported receipt format")
            print("3. Items not in expected format")
            print("\nTry:")
            print("- Taking a clearer photo with even lighting")
            print("- Cropping to just the items section")
            print("- Checking the raw OCR output above for detection issues")
    else:
        print(f"❌ Error processing receipt: {result['error']}")