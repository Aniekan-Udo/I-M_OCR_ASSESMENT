import streamlit as st
import re
import math
import json
import os
import warnings
from pathlib import Path
from PIL import Image
import numpy as np
from io import BytesIO

# CRITICAL: Set environment BEFORE any Paddle imports
os.environ['FLAGS_enable_mkldnn'] = 'False'
os.environ['PADDLE_DISABLE_SIGNAL_HANDLER'] = '1'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Suppress warnings
warnings.filterwarnings('ignore', message='.*tesseract.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Page config
st.set_page_config(
    page_title="Product Extractor", 
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global storage for extracted products
if 'EXTRACTED_PRODUCTS' not in st.session_state:
    st.session_state.EXTRACTED_PRODUCTS = []

# Fixed OCR loader - REMOVED conflicting parameters
@st.cache_resource
def load_ocr():
    """Load PaddleOCR - Fixed parameter conflict"""
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en", 
        use_angle_cls=True, 
        det_db_thresh=0.3,      # Use ONE threshold parameter
        det_db_box_thresh=0.5,
        cpu_threads=1,
        enable_mkldnn=False
    )

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

price_pattern = re.compile(r"(?:[$€£]\s*)?(\d{1,2}\.\d{2}|\d{1,2})\b")
size_pattern = re.compile(r"\b\d+\s*(g|kg|ml|l)\b|\b(pack|pk)\b", re.I)

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

def run_ocr(image):
    ocr = load_ocr()
    
    if isinstance(image, str):
        raw = ocr.ocr(image)
    else:
        raw = ocr.ocr(np.array(image))
    
    if not isinstance(raw, list) or len(raw) == 0:
        return []
    
    page = raw[0]
    rec_texts = page.get("rec_texts", [])
    rec_scores = page.get("rec_scores", [])
    rec_boxes = page.get("rec_boxes", [])
    
    lines = []
    for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
        x1, y1, x2, y2 = map(float, box)
        lines.append({
            "text": str(text),
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
        })
    return lines

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
        
        groups.append({
            "price_line": p,
            "context_lines": related,
            "bbox": prod_box,
        })
    
    return groups

def choose_product_name(price_line, context_lines):
    px1, py1, px2, py2 = price_line["bbox"]
    py_bottom = py2  
    
    candidates = []
    for l in context_lines:
        if is_price(l["text"]):
            continue
        
        x1, y1, x2, y2 = l["bbox"]
        
        if y1 < py_bottom - 20:
            continue
        
        text = l["text"].strip()
        if sum(c.isalpha() for c in text) < 3:
            continue
        
        candidates.append(l)
    
    if not candidates:
        return ""
    
    def score(l):
        x1, y1, x2, y2 = l["bbox"]
        dist_y = abs(y1 - py_bottom)
        length_score = len(l["text"])
        return length_score - 0.2 * dist_y
    
    best = max(candidates, key=score)
    return best["text"]

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

def extract_products(image):
    lines = run_ocr(image)
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

def save_uploaded_file(uploaded_file):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    filepath = os.path.join(uploads_dir, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

def main():
    st.title("🛒 Leaflet Product Extractor")
    st.markdown("Upload a supermarket leaflet to automatically extract product names, prices, and details")
    
    st.sidebar.header("📁 Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
        help="Supports JPG, PNG, WebP (max 16MB)"
    )
    
    col1, col2 = st.columns([2, 1])
    
    if uploaded_file is not None and allowed_file(uploaded_file.name):
        filepath = save_uploaded_file(uploaded_file)
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Uploaded Leaflet", use_column_width=True)
            st.info(f"📐 Image: {image.size[0]}x{image.size[1]} pixels")
        
        if st.button("🔍 Extract Products", type="primary"):
            with st.spinner("Running OCR + Product extraction..."):
                try:
                    st.session_state.EXTRACTED_PRODUCTS = extract_products(image)
                except Exception as e:
                    st.error(f"OCR failed: {str(e)}")
                    st.session_state.EXTRACTED_PRODUCTS = []
            
            if st.session_state.EXTRACTED_PRODUCTS:
                st.success(f"✅ Extracted {len(st.session_state.EXTRACTED_PRODUCTS)} products!")
                
                with col2:
                    st.subheader("📊 Products")
                
                df_data = [{"ID": p["id"], 
                          "Product": p["product_name"][:50] + "..." if len(p["product_name"]) > 50 else p["product_name"],
                          "Price": f"${p['price']:.2f}" if p['price'] else "N/A",
                          "Size": p.get("size", ""),
                          "Promo": p.get("promo_text", "")[:30] + "..." if p.get("promo_text") else ""}
                         for p in st.session_state.EXTRACTED_PRODUCTS]
                
                st.dataframe(df_data, use_container_width=True, hide_index=True)
                
                with st.expander(f"👁️ View all {len(st.session_state.EXTRACTED_PRODUCTS)} products (JSON)"):
                    st.json(st.session_state.EXTRACTED_PRODUCTS)
                
                json_data = json.dumps(st.session_state.EXTRACTED_PRODUCTS, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download products.json",
                    data=json_data,
                    file_name="extracted_products.json",
                    mime="application/json",
                    type="secondary"
                )
            else:
                st.warning("❌ No products found. Try a clearer image with visible prices.")
    
    elif uploaded_file is not None:
        st.error("❌ Invalid file type. Use JPG, PNG, GIF, BMP, or WebP.")

if __name__ == "__main__":
    main()
