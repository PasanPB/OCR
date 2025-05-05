from flask import Flask, request, jsonify
from bill_scanner import scan_and_categorize

app = Flask(__name__)

@app.route('/scan', methods=['POST'])
def scan_bill():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image_file = request.files['file']
    result = scan_and_categorize(image_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
