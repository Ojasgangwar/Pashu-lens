# streamlit_app.py
#streamlit run streamlit_app.py

# 🐄 Cattle & Buffalo Breed Recognition


import os
import pickle
import re
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import base64
import json
import streamlit.components.v1 as components


# Config & CSS

st.set_page_config(page_title="Cattle & Buffalo Breed Classifier", layout="wide")
st.markdown("<style>#MainMenu{visibility:hidden;} footer{visibility:hidden;}</style>", unsafe_allow_html=True)

BASE_CSS = """
<style>
:root{
  --bg:#0E1117;
  --muted:#8892a0;
  --accent:#FFA500; 
}
body, .stApp, .main, .block-container { background-color: var(--bg) !important; color: #E6EDF3 !important; }
.app-title{ font-size:34px; font-weight:700; color:white; }
.helper-bar{ background:#203245; padding:10px; border-radius:6px; color:#d9eefc; margin-top:8px; font-size:18px; }
.info-card, .pred-card, .breed-detail-card { font-size:18px; padding:18px; border-radius:10px; }
.pred-card { background:linear-gradient(180deg,#161821,#0e1116); }
.pdf-placeholder{ height:180px; border-radius:6px; background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00)); display:flex; align-items:center; justify-content:center; color:#9fb4d7; font-size:18px; }
.stMetricValue { font-size:36px !important; font-weight:700; color:#FFA500 !important; }
.stMetricLabel { font-size:22px !important; font-weight:600; }
.stExpanderHeader { font-size:20px !important; font-weight:600; }
.status-bar{ background:#142533; color:#d6eefc; padding:12px; border-radius:6px; margin-top:18px; font-size:18px; }
.small-muted{ color:#9aa7b7; font-size:15px; }
.card-title{ font-size:20px; font-weight:700; margin-bottom:6px; }
.top3-card{ background:#0a0c10; padding:10px; border-radius:8px; text-align:center; margin-bottom:6px; font-size:16px; }
.breed-detail-card{ background:#1A1C22; padding:14px; border-radius:8px; margin-top:6px; font-size:16px; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


# Header

st.markdown("""
<div style='display:flex; gap:12px; align-items:center;'>
    <div style='font-size:34px;'>🐄</div>
    <div>
        <div class='app-title'>Cattle & Buffalo Breed Classifier</div>
        <div style='color:var(--muted); margin-top:2px;'>
            Upload images (main image first). Predictions, PDF preview, metrics, breed details.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Model & Settings

with st.sidebar:
    st.markdown("## Model & Settings")
    keras_model_path = st.text_input("Keras model path", value="model.h5")
    if st.button("Load model"):
        st.success("Model load placeholder (no-op in UI).")
    st.markdown("---")
    img_target_size = st.selectbox("Image target size", options=["(224,224)", "(160,160)"], index=0)
    st.markdown("---")
    similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.markdown("---")
    st.markdown("### Class labels:")
    show_state = st.checkbox("Show session state", value=False)


# Paths & class list

DATA_DIR = "data/"
PDF_FOLDER = "pdf/"
CACHE_PATH = "cache.pkl"
BREED_INFO_FILE = "breed_info.json"

CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]) if os.path.exists(DATA_DIR) else []

with st.sidebar:
    if CLASS_NAMES:
        for cls in CLASS_NAMES:
            st.markdown(f"<div style='color:var(--accent); font-weight:600'>{cls}</div>", unsafe_allow_html=True)
    else:
        st.info("No classes found in data/")

# Load feature model & cache

@st.cache_resource
def load_feature_model():
    
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
feature_model = load_feature_model()

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        dataset_cache = pickle.load(f)
else:
    dataset_cache = None
    st.sidebar.error("Cache not found. Run prepare_cache.py first.")

# Load breed info database

if os.path.exists(BREED_INFO_FILE):
    with open(BREED_INFO_FILE, "r", encoding="utf-8") as f:
        breed_info = json.load(f)
else:
    breed_info = {}

# Helper functions

def preprocess_for_mobilenet(img, target_size=(224,224)):
    img = img.convert("RGB").resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def get_embedding(img):
    x = preprocess_for_mobilenet(img)
    return feature_model.predict(x).flatten()

def safe_float(val, default=None):
    """
    Parse numeric value from strings such as:
      - "≈120 cm" -> 120.0
      - "350–400 kg" -> average -> 375.0
      - "≈4.5%" -> 4.5
    Return default if not parseable.
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
    
        val_clean = str(val).replace("≈", "").replace("%", "")
   
        nums = re.findall(r"[\d.]+", val_clean)
        if not nums:
            return default
        nums = [float(n) for n in nums]
        return sum(nums) / len(nums)
    except Exception:
        return default



def display_breed_pdf(breed_name):
    """
    Shows a small inline PDF preview, a Download button and an "Expand PDF" button.
    When Expand is clicked, a modal opens containing the full PDF and a working Close button.
    This uses st.modal (if available) or a fallback container.
    """
    pdf_file = f"{breed_name}.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    if not os.path.exists(pdf_path):
        st.info(f"No detailed PDF available for {breed_name}.")
        return

    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe_preview_html = f'<iframe src="data:application/pdf;base64,{base64_pdf}#view=FitH" width="100%" height="100%" style="border:none;"></iframe>'

    st.markdown("#### 📖 Breed Details")
  
    st.markdown(f'<div class="pdf-placeholder">{iframe_preview_html}</div>', unsafe_allow_html=True)

 
    key_name = f"modal_open_{breed_name}"
    if key_name not in st.session_state:
        st.session_state[key_name] = False

    colA, colB = st.columns([1, 1])
    with colA:
        st.download_button("⬇ Download PDF", data=pdf_bytes, file_name=pdf_file, mime="application/pdf")
    with colB:
     
        if not st.session_state[key_name]:
            if st.button("🔎 Expand PDF", key=f"expand_{breed_name}"):
                st.session_state[key_name] = True

  
    if st.session_state[key_name]:
       
        if hasattr(st, "modal"):
          
            try:
                with st.modal(f"{breed_name} — PDF", clear_on_submit=False):
                    # show a large embedded PDF inside the modal using components.html
                    components.html(f'<iframe src="data:application/pdf;base64,{base64_pdf}#view=FitH" width="100%" height="800" style="border:none;"></iframe>',
                                    height=820)
                  
                    if st.button("❌ Close PDF", key=f"close_modal_{breed_name}"):
                        st.session_state[key_name] = False
                return
            except Exception:
               
                pass

        
        placeholder = st.empty()
        with placeholder.container():
            # overlay wrapper
            st.markdown(
                """
                <div style="
                    position: fixed; top:0; left:0;
                    width:100%; height:100%;
                    background-color: rgba(0,0,0,0.85);
                    display:flex; justify-content:center; align-items:center; z-index:9999;">
                    <div style="position: relative; width: 90%; height: 90%; max-width:1100px; background:white; border-radius:10px; overflow:hidden;">
                """,
                unsafe_allow_html=True,
            )
            # top row: close button (Streamlit button below will appear under the overlay, but we'll render it now)
            # embed the PDF via components.html (this will render inside the placeholder DOM)
            components.html(f'<iframe src="data:application/pdf;base64,{base64_pdf}#view=FitH" width="100%" height="100%" style="border:none;"></iframe>',
                            height=800)
            # show a visible Streamlit button under the iframe container (it will appear after iframe)
            if st.button("❌ Close PDF", key=f"close_modal_fallback_{breed_name}"):
                st.session_state[key_name] = False
            # end overlay wrapper
            st.markdown("</div></div>", unsafe_allow_html=True)

        # if modal closed, clear placeholder on next run
        if not st.session_state[key_name]:
            placeholder.empty()


# Upload & prediction area

st.markdown("<div class='helper-bar'>Upload images (main image first). Accepts PNG/JPG/JPEG/PDF.</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    # choose first image-like file as main
    main_file = next((f for f in uploaded_files if f.type.startswith("image/")), uploaded_files[0])
    img = Image.open(main_file).convert("RGB")

    if dataset_cache is None:
        st.error("Dataset cache missing. Cannot compute predictions.")
        st.stop()

    # compute embedding and similarities
    with st.spinner("Analyzing image..."):
        img_feat = get_embedding(img)
        # dataset_cache is expected to be dict: {class_name: array_of_feature_vectors}
        similarities = {
            cls: float(np.mean(cosine_similarity([img_feat], np.stack(feats))))
            for cls, feats in dataset_cache.items()
        }

    sorted_breeds = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    pred_class, pred_score = sorted_breeds[0]

    if pred_score < similarity_threshold:
        st.error("The uploaded image does not appear to be a cattle or buffalo. Please upload a relevant image.")
        st.stop()

    # Main image display
    st.markdown("<h3>Main image</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img, use_container_width=True)

    st.markdown("<hr style='border:none; height:1px; background:rgba(255,255,255,0.04); margin-top:18px; margin-bottom:18px;' />",
                unsafe_allow_html=True)

    # Results & Report region
    st.markdown("<h3>Results & Report</h3>", unsafe_allow_html=True)
    rcol1, rcol2, rcol3 = st.columns([1, 1, 1])

    # Predicted Breed card
    with rcol1:
        st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>Predicted breed</div>"
                    f"<div style='font-size:26px; font-weight:800; color:#FFA500'>{pred_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Confidence: {pred_score*100:.2f}%</div>", unsafe_allow_html=True)

        # show representative sample image from dataset if available
        try:
            imgs_list = os.listdir(os.path.join(DATA_DIR, pred_class))
            pic_path = next((os.path.join(DATA_DIR, pred_class, f)
                             for f in imgs_list if f.lower().endswith((".jpg", ".jpeg", ".png"))), None)
            if pic_path:
                thumb = Image.open(pic_path)
                thumb.thumbnail((300, 180))
                st.image(thumb, use_container_width=True)
            else:
                st.markdown("<div style='height:120px; background:#0C0F13; border-radius:6px; display:flex; align-items:center; justify-content:center;'>No example image</div>",
                            unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("</div>", unsafe_allow_html=True)

    # PDF preview & modal open
    with rcol2:
        display_breed_pdf(pred_class)

    # Breed Metrics (Height / Weight / Milk Fat %) — show JSON original string (if present) and compute progress
    with rcol3:
        st.markdown("<div class='info-card'><div class='card-title'>Breed Metrics</div>", unsafe_allow_html=True)
        info = breed_info.get(pred_class, {})

        # raw strings (keep original display) and numeric parsing for progress
        raw_height = info.get("Height", None)
        raw_weight = info.get("Weight", None)
       # raw_milk = info.get("Milk Fat %", None)

        height_val = safe_float(raw_height)
        weight_val = safe_float(raw_weight)
      #  milk_val = safe_float(raw_milk)

        # scales set by user: height 50-200 cm, weight 150-800 kg, milk fat 2-12 %
        h_min, h_max = 50.0, 200.0
        w_min, w_max = 150.0, 800.0
        m_min, m_max = 2.0, 12.0

        # Height
        if height_val is not None:
            h_prog = int(np.clip((height_val - h_min) / (h_max - h_min) * 100, 0, 100))
            disp_height = raw_height if raw_height is not None else f"{height_val}"
            st.metric("Height (cm)", f"{disp_height}", f"{h_prog}%")
            st.progress(h_prog)
        else:
            st.metric("Height (cm)", raw_height if raw_height else "N/A", "")

        # Weight
        if weight_val is not None:
            w_prog = int(np.clip((weight_val - w_min) / (w_max - w_min) * 100, 0, 100))
            disp_weight = raw_weight if raw_weight is not None else f"{weight_val}"
            st.metric("Weight (kg)", f"{disp_weight}", f"{w_prog}%")
            st.progress(w_prog)
        else:
            st.metric("Weight (kg)", raw_weight if raw_weight else "N/A", "")

        # Milk Fat %
       # if milk_val is not None:
        #    m_prog = int(np.clip((milk_val - m_min) / (m_max - m_min) * 100, 0, 100))
         #   disp_milk = (f"{milk_val:.1f}%") if raw_milk is None else raw_milk
          #  st.metric("Milk Fat %", f"{disp_milk}", f"{m_prog}%")
           # st.progress(m_prog)
        #else:
         #   st.metric("Milk Fat %", raw_milk if raw_milk else "N/A", "")

        st.markdown("</div>", unsafe_allow_html=True)

    # Breed Details Panel (main breed only)
    st.markdown("<h3 style='margin-top:24px;'>Breed Details</h3>", unsafe_allow_html=True)
    with st.expander(f"🔹 {pred_class}", expanded=True):
        info = breed_info.get(pred_class, None)
        if info:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Height:** <span style='font-size:18px; color:#FFA500'>{info.get('Height','N/A')}</span>", unsafe_allow_html=True)
                st.markdown(f"**Weight:** <span style='font-size:18px; color:#FFA500'>{info.get('Weight','N/A')}</span>", unsafe_allow_html=True)
                st.markdown(f"**Location:** <span style='font-size:18px'>{info.get('Location','N/A')}</span>", unsafe_allow_html=True)
                st.markdown(f"**Color:** <span style='font-size:18px'>{info.get('Color','N/A')}</span>", unsafe_allow_html=True)
            with col2:
              #  st.markdown(f"**Milk Fat %:** <span style='font-size:18px; color:#FFA500'>{info.get('Milk Fat %','N/A')}</span>", unsafe_allow_html=True)
                st.markdown(f"**Horn Shape & Size:** <span style='font-size:18px'>{info.get('Horn Shape & Size','N/A')}</span>", unsafe_allow_html=True)
                st.markdown(f"**Visible Characteristics:** <span style='font-size:18px'>{info.get('Visible Characteristics','N/A')}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#9aa7b7;'>No detailed info available.</div>", unsafe_allow_html=True)

    # Similar Breeds (show 4th-6th to avoid showing exact predicted class first)
    similar_breeds_for_display = sorted_breeds[3:6] if len(sorted_breeds) >= 6 else sorted_breeds[1:4]
    st.markdown("<h3 style='margin-top:18px;'>Similar Breeds</h3>", unsafe_allow_html=True)
    for bname, bscore in similar_breeds_for_display:
        with st.expander(f"🔹 {bname} ({bscore*100:.2f}%)", expanded=False):
            # show sample image if available
            img_path = next((os.path.join(DATA_DIR, bname, f) for f in os.listdir(os.path.join(DATA_DIR, bname))
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))), None) if os.path.exists(os.path.join(DATA_DIR, bname)) else None
            if img_path:
                sim_img = Image.open(img_path)
                sim_img.thumbnail((420, 260))
                st.image(sim_img, use_container_width=True)
            else:
                st.markdown("<div style='height:160px; background:#0C0F13; border-radius:6px; display:flex; align-items:center; justify-content:center;'>No image</div>",
                            unsafe_allow_html=True)

            sinfo = breed_info.get(bname, {})
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Height:** {sinfo.get('Height','N/A')}")
                st.markdown(f"**Weight:** {sinfo.get('Weight','N/A')}")
                st.markdown(f"**Location:** {sinfo.get('Location','N/A')}")
                st.markdown(f"**Color:** {sinfo.get('Color','N/A')}")
            with col2:
              #  st.markdown(f"**Milk Fat %:** {sinfo.get('Milk Fat %','N/A')}")
                st.markdown(f"**Horn Shape & Size:** {sinfo.get('Horn Shape & Size','N/A')}")
                st.markdown(f"**Visible Characteristics:** {sinfo.get('Visible Characteristics','N/A')}")

    st.markdown(f"<div class='status-bar'>Dataset classes: {len(CLASS_NAMES)}</div>", unsafe_allow_html=True)

# Optional session state dump (debugging)
if show_state:
    st.markdown("### Session state")
    st.write(dict(st.session_state))
