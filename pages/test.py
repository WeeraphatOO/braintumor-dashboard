import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="YOLO MRI Segmentation",
    page_icon="🧠",
    layout="centered"
)

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    return YOLO(r"..\models\3cls\direct\best_yolo_direct_3cls.pt")

model = load_model()

# =========================
# UI
# =========================
st.title("Brain MRI YOLO Segmentation")
st.write("Upload a brain MRI image to run YOLO segmentation.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# INFERENCE
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Input Image")
    st.image(image, width="stretch")

    results = model(img_np, conf=0.25, verbose=False)
    result = results[0]
    annotated = result.plot()

    st.subheader("YOLO Segmentation Result")
    st.image(annotated, width="stretch")

    if result.boxes is not None and len(result.boxes) > 0:
        st.success(f"Detected {len(result.boxes)} object(s)")
    else:
        st.warning("No objects detected")
