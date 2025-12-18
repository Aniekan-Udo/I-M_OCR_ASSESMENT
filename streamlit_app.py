import re
import math
import json
import warnings
import tempfile
import os
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', message='.*tesseract.*')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR

# Page config
st.set_page_config(
    page_title="Leaflet Product Extractor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .product-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .price-tag {
        font-size: 1.5rem;
        font-weight: bold;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OCR (cached to avoid reloading)
@st.cache_resource
def init_ocr():
    return PaddleOCR(lang="en", use_angle_cls=True)

ocr = init_ocr()

price_pattern = re.compile(r"(?:[$€£]\s*)?(\d{1,2}\.\d{1,2}|\d{1,2})\b")
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

def run_ocr(image_path: str):
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
    py_center = 0.5 * (py1 + py2)

    candidates = []
    for l in context_lines:
        if is_price(l["text"]):
            continue
        x1, y1, x2, y2 = l["bbox"]
        if y1 > py_center + 10:
            continue
        text = l["text"].strip()
        if sum(c.isalpha() for c in text) < 3:
            continue
        candidates.append(l)

    if not candidates:
        return ""

    def score(l):
        x1, y1, x2, y2 = l["bbox"]
        center_y = 0.5 * (y1 + y2)
        ideal_y = py_center - 20
        dist_y = abs(center_y - ideal_y)
        return len(l["text"]) - 0.05 * dist_y

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
# Streamlit UI
# -----------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🛒 Leaflet Product Extractor</h1>
    <p>Intelligent OCR-based product information extraction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.write("""
    This app extracts product information from grocery leaflet images using OCR technology.
    
    **Features:**
    - 📸 Upload leaflet images
    - 🔍 Automatic product detection
    - 💰 Price extraction
    - 📦 Size & promotion detection
    - 📥 Export to JSON
    """)
    
    st.divider()
    
    st.header("📖 Instructions")
    st.write("""
    1. Upload a leaflet image
    2. Wait for processing (10-30s)
    3. View extracted products
    4. Download JSON if needed
    """)

# Main content
uploaded_file = st.file_uploader(
    "Upload Leaflet Image",
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
    help="Upload a grocery leaflet image for product extraction"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded leaflet image", use_column_width=True)
    
    with col2:
        st.subheader("ℹ️ Image Info")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
    
    # Process button
    if st.button("🚀 Extract Products", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing image with OCR... This may take 10-30 seconds..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Extract products
                products = extract_products(tmp_path)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                # Store in session state
                st.session_state['products'] = products
                
                st.success(f"✅ Successfully extracted {len(products)} products!")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")

# Display results if products exist
if 'products' in st.session_state and st.session_state['products']:
    products = st.session_state['products']
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(products)}</div>
            <div class="stat-label">Total Products</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = sum(p['price'] for p in products) / len(products) if products else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">${avg_price:.2f}</div>
            <div class="stat-label">Average Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        promo_count = sum(1 for p in products if p.get('promo_text'))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{promo_count}</div>
            <div class="stat-label">Promotions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search products", placeholder="Type to search...")
    with col2:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(products, indent=2),
            file_name="extracted_products.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Filter products
    filtered_products = products
    if search_term:
        filtered_products = [
            p for p in products 
            if search_term.lower() in p.get('product_name', '').lower() 
            or search_term.lower() in p.get('promo_text', '').lower()
        ]
    
    st.subheader(f"🛍️ Products ({len(filtered_products)} found)")
    
    # Display products in cards or table
    view_mode = st.radio("View Mode", ["Table", "Cards"], horizontal=True)
    
    if view_mode == "Table":
        # Table view
        if filtered_products:
            df = pd.DataFrame([{
                'ID': p['id'],
                'Product Name': p.get('product_name', 'N/A'),
                'Price': f"${p.get('price', 0):.2f}",
                'Size': p.get('size', '-'),
                'Unit Price': p.get('unit_price_text', '-'),
                'Promotion': p.get('promo_text', '-')
            } for p in filtered_products])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No products found matching your search.")
    
    else:
        # Card view
        if filtered_products:
            for product in filtered_products:
                with st.expander(f"#{product['id']} - {product.get('product_name', 'Unknown Product')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Price:** <span class='price-tag'>${product.get('price', 0):.2f}</span>", unsafe_allow_html=True)
                        st.write(f"**Price Unit:** {product.get('price_unit', 'N/A')}")
                        st.write(f"**Size:** {product.get('size', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Unit Price:** {product.get('unit_price_text', 'N/A')}")
                        if product.get('promo_text'):
                            st.success(f"🎉 {product['promo_text']}")
                        st.write(f"**Bounding Box:** [{', '.join([f'{v:.1f}' for v in product['bbox']])}]")
        else:
            st.info("No products found matching your search.")

else:
    # Initial state - show instructions
    st.info("👆 Upload a leaflet image to get started!")
    
    st.subheader("📝 Supported Formats")
    st.write("PNG, JPG, JPEG, GIF, BMP, WebP")
    
    st.subheader("💡 Tips for Best Results")
    st.write("""
    - Use clear, well-lit images
    - Ensure text is readable
    - Avoid excessive blur or distortion
    - Higher resolution images work better
    """)
