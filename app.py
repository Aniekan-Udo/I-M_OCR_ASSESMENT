import re
import math
import json
import sys
import os
import warnings
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename

# Suppress warnings
warnings.filterwarnings('ignore', message='.*tesseract.*')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR

# -----------------------------
# Flask App Setup
# -----------------------------
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# FIX: Use absolute path for uploads folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# FIX: Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Global storage for extracted products
EXTRACTED_PRODUCTS = []

# -----------------------------
# Init OCR (CPU, modern flags)
# -----------------------------
ocr = PaddleOCR(
    lang="en", 
    use_textline_orientation=True, 
    text_det_thresh=0.3,  # Detects more text boxes
    text_det_box_thresh=0.5
)

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Stage 1: OCR → normalized lines
# -----------------------------
def run_ocr(image_path: str):
    """
    Uses high-level PaddleOCR output
    """
    raw = ocr.ocr(image_path)
    if not isinstance(raw, list) or len(raw) == 0:
        return []
    
    page = raw[0]
    rec_texts = page.get("rec_texts", [])
    rec_scores = page.get("rec_scores", [])
    rec_boxes = page.get("rec_boxes", [])
    
    lines = []
    for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
        x1, y1, x2, y2 = map(float, box)
        lines.append(
            {
                "text": str(text),
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
            }
        )
    return lines

# -----------------------------
# Helpers
# -----------------------------
price_pattern = re.compile(
    r"(?:[$€£]\s*)?(\d{1,2}\.\d{1,2}|\d{1,2})\b"
)

size_pattern = re.compile(
    r"\b\d+\s*(g|kg|ml|l)\b|\b(pack|pk)\b", re.I
)

def is_price(text: str) -> bool:
    t = text.lower()
    bad_tokens = ["mm", "×", "x", "g ", "g/", "ml", "l ", "pack", "pk", "per 100g", "per kg"]
    if any(bt in t for bt in bad_tokens):
        return False
    
    if "$" in t or "€" in t or "£" in t:
        return bool(price_pattern.search(text))
    
    if len(t) <= 6:
        return bool(price_pattern.search(text))
    
    return False

def parse_price(text: str):
    m = price_pattern.search(text)
    if not m:
        return None
    val = m.group(1)
    try:
        return float(val)
    except ValueError:
        return None

def center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

# -----------------------------
# Stage 2a: group lines by product
# -----------------------------
def group_products(lines, max_dist: float = 250.0):
    prices = [l for l in lines if is_price(l["text"])]
    others = [l for l in lines if not is_price(l["text"])]
    
    groups = []
    for p in prices:
        cx, cy = center(p["bbox"])
        related = []
        
        for l in others:
            lx, ly = center(l["bbox"])
            d = math.dist((cx, cy), (lx, ly))
            if d <= max_dist:
                related.append(l)
        
        if not related:
            continue
        
        all_boxes = [p["bbox"]] + [l["bbox"] for l in related]
        xs = [b[0] for b in all_boxes] + [b[2] for b in all_boxes]
        ys = [b[1] for b in all_boxes] + [b[3] for b in all_boxes]
        prod_box = [min(xs), min(ys), max(xs), max(ys)]
        
        groups.append(
            {
                "price_line": p,
                "context_lines": related,
                "bbox": prod_box,
            }
        )
    
    return groups

# -----------------------------
# Product-name heuristic - FIXED TO LOOK BELOW PRICE
# -----------------------------
def choose_product_name(price_line, context_lines):
    px1, py1, px2, py2 = price_line["bbox"]
    py_center = 0.5 * (py1 + py2)
    py_bottom = py2  # Bottom of price badge
    
    candidates = []
    for l in context_lines:
        if is_price(l["text"]):
            continue
        
        x1, y1, x2, y2 = l["bbox"]
        
        # FIXED: Look for text BELOW the price (y1 > py_bottom)
        # Allow some overlap (y1 > py_bottom - 20)
        if y1 < py_bottom - 20:
            continue
        
        text = l["text"].strip()
        
        # Must have at least 3 letters
        if sum(c.isalpha() for c in text) < 3:
            continue
        
        candidates.append(l)
    
    if not candidates:
        return ""
    
    # Score candidates: prefer text immediately below price
    def score(l):
        x1, y1, x2, y2 = l["bbox"]
        center_y = 0.5 * (y1 + y2)
        
        # Prefer text closer to price bottom
        dist_y = abs(y1 - py_bottom)
        
        # Prefer longer text (likely product name)
        length_score = len(l["text"])
        
        # Penalize distance heavily
        return length_score - 0.2 * dist_y
    
    best = max(candidates, key=score)
    return best["text"]

# -----------------------------
# Stage 2b: extract fields
# -----------------------------
def extract_fields(group):
    price_text = group["price_line"]["text"]
    price = parse_price(price_text)
    
    price_unit = None
    lp = price_text.lower()
    if "kg" in lp:
        price_unit = "kg"
    elif "each" in lp or " ea" in lp:
        price_unit = "each"
    elif "pack" in lp or " pk" in lp:
        price_unit = "pack"
    
    candidates = [l for l in group["context_lines"] if not is_price(l["text"])]
    
    product_name = choose_product_name(group["price_line"], candidates)
    
    size = ""
    unit_price_text = ""
    promo_text = ""
    
    for l in candidates:
        t = l["text"]
        tl = t.lower()
        
        if not size and size_pattern.search(t):
            size = t
        
        if "per " in tl:
            unit_price_text = t
        
        if any(k in tl for k in ["every day", "on sale", "until", "buy ", "save"]):
            promo_text = t
    
    return {
        "product_name": product_name,
        "price": price,
        "price_unit": price_unit,
        "size": size,
        "unit_price_text": unit_price_text,
        "promo_text": promo_text,
        "bbox": group["bbox"],
        "raw_price_text": price_text,
    }

# -----------------------------
# Stage 3: high-level API with cleaning
# -----------------------------
def extract_products(image_path: str):
    lines = run_ocr(image_path)
    groups = group_products(lines)
    products = [extract_fields(g) for g in groups]
    
    cleaned = []
    for idx, p in enumerate(products):
        if p["price"] is None:
            continue
        
        if sum(c.isalpha() for c in p["product_name"]) < 3:
            continue
        
        if p["price"] > 100 and "$" not in p["raw_price_text"]:
            continue
      
        p["id"] = idx + 1
        cleaned.append(p)
    
    return cleaned

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle image upload and process with OCR"""
    global EXTRACTED_PRODUCTS
  
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
  
    file = request.files['file']
  
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
  
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type. Allowed: JPG, PNG, GIF, WebP'}), 400
  
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
      
        file.save(filepath)
      
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': f'Failed to save file at {filepath}'}), 500
      
        print(f"Processing {filename} at {filepath}...")
        EXTRACTED_PRODUCTS = extract_products(filepath)
      
        return jsonify({
            'success': True,
            'message': f'Successfully extracted {len(EXTRACTED_PRODUCTS)} products',
            'product_count': len(EXTRACTED_PRODUCTS),
            'products': EXTRACTED_PRODUCTS
        })
  
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500

@app.route('/api/products')
def get_products():
    """Return all extracted products as JSON"""
    return jsonify(EXTRACTED_PRODUCTS)

@app.route('/api/product/<int:product_id>')
def get_product(product_id):
    """Return specific product details"""
    for product in EXTRACTED_PRODUCTS:
        if product.get('id') == product_id:
            return jsonify({
                'success': True,
                'product': product,
                'message': f'Selected product: {product["product_name"]}'
            })
    return jsonify({'success': False, 'message': 'Product not found'}), 404

@app.route('/api/download')
def download_json():
    """Download products as JSON file"""
    if not EXTRACTED_PRODUCTS:
        return jsonify({'success': False, 'message': 'No products to export'}), 400
  
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.json')
  
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(EXTRACTED_PRODUCTS, f, indent=2, ensure_ascii=False)
  
    return send_file(output_path, as_attachment=True, download_name='extracted_products.json')

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("Leaflet Product Extractor - Web Application")
    print("="*60)
    print(f"\n✓ Upload folder: {UPLOAD_FOLDER}")
    print("✓ Upload images through the web interface")
    print("✓ Extract product information automatically")
    print("✓ Download results as JSON")
    print("\n" + "="*60)
    print("Starting web server...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*60 + "\n")
  
    app.run(debug=True, host='127.0.0.1', port=5000)