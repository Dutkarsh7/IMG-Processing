"""
Smart Document Scanner - Web Interface
"""

import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# Try to find Tesseract in common Windows locations
tesseract_paths = [
    r'C:\Program Files\Autopsy-4.22.1\autopsy\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    rf'C:\Users\{os.getenv("USERNAME")}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
]

for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"✓ Found Tesseract at: {path}")
        break
else:
    print("⚠ Tesseract not found in common locations. Please set path manually.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process uploaded image"""
    # Load and resize
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Save processed images
    cv2.imwrite('outputs/grayscale.jpg', gray)
    cv2.imwrite('outputs/threshold.jpg', thresh)
    
    return gray, thresh

def extract_text(image):
    """Extract text using Tesseract OCR"""
    try:
        # Check if Tesseract is available
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        
        # Extract text with config
        text = pytesseract.image_to_string(image, config='--psm 6')
        
        if not text.strip():
            return "No text detected in the image"
        
        return text
    except pytesseract.TesseractNotFoundError:
        return "ERROR: Tesseract not installed. Install from: https://github.com/UB-Mannheim/tesseract/wiki"
    except Exception as e:
        return f"OCR Error: {str(e)}"

def image_to_base64(image_path):
    """Convert image to base64 for display"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        gray, thresh = process_image(filepath)
        
        # Extract text
        text = extract_text(gray)
        
        # Convert images to base64
        original_b64 = image_to_base64(filepath)
        gray_b64 = image_to_base64('outputs/grayscale.jpg')
        thresh_b64 = image_to_base64('outputs/threshold.jpg')
        
        return jsonify({
            'success': True,
            'text': text,
            'images': {
                'original': original_b64,
                'grayscale': gray_b64,
                'threshold': thresh_b64
            }
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f'outputs/{filename}', as_attachment=True)

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=5000)
