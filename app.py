from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from src.features import embed_image, get_device, load_backbone

MODEL_PATH = Path("models/dinov2_base.joblib")

def display_name(class_name):
    # "016.Painted_Bunting" -> "Painted Bunting"
    return class_name.split(".", 1)[-1].replace("_", " ")

st.set_page_config(page_title="Bird Species Identifier", page_icon=None, layout="centered")
st.title("Bird Species Identification")
st.write("Upload a bird photograph to predict its species using the trained CUB-200-2011 classifier.")

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the model first. Expected {MODEL_PATH} (run train_features.py).")
    saved = joblib.load(MODEL_PATH)
    device = get_device()
    backbone = load_backbone(saved["backbone"], device)
    return backbone, device, saved["classifier"], saved["class_names"]

uploaded = st.file_uploader("Choose a bird image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    try:
        backbone, device, classifier, class_names = load_assets()
        probabilities = classifier.predict_proba(embed_image(backbone, image, device))[0]
        top_indices = np.argsort(probabilities)[-5:][::-1]
        best = int(top_indices[0])
        st.subheader("Prediction")
        st.write(f"**{display_name(class_names[best])}**")
        st.caption("Scores are model outputs and are not guaranteed to be calibrated probabilities.")
        st.subheader("Top five predictions")
        for idx in top_indices:
            st.write(f"{display_name(class_names[int(idx)])} — {probabilities[idx] * 100:.2f}%")
            st.progress(float(probabilities[idx]))
    except Exception as exc:
        st.error(str(exc))
