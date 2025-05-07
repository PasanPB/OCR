from flask import Flask, request, jsonify
from pymongo import MongoClient
import os
from bill_scanner import RestaurantBillScanner

app = Flask(__name__)

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["finance_app"]
collection = db["bills"]

# Upload folder
UPLOAD_FOLDER = 'test_bills'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize your scanner class
scanner = RestaurantBillScanner()

@app.route('/scan_bill', methods=['POST'])
def scan_bill():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Use your scanner class
    result = scanner.process_bill(file_path)

    if result['status'] == 'success':
        # Insert into MongoDB
        collection.insert_one(result)
        return jsonify(result), 200
    else:
        return jsonify(result), 500

if __name__ == '__main__':
    app.run(debug=True)
