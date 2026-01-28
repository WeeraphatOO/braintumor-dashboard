import streamlit as st
from PIL import Image
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Input MRI Scan",
    page_icon="🎈",
    layout="wide"
)

# =========================
# LOAD YOLO MODEL (SAFE)
# =========================
@st.cache_resource
def load_yolo_seg():
    from ultralytics import YOLO
    return YOLO("models/3cls/direct/best_yolo_direct_3cls.pt")

# =========================
# CENTER LAYOUT
# =========================
left, center, right = st.columns([1, 2, 1])

with center:
    st.title("Input Your MRI Scan!")

    st.markdown(
        "You can upload a brain MRI scan and select a model "
        "to perform tumor detection and segmentation."
    )

    st.write("1. U-Net (Semantic Segmentation Model)")
    st.write("2. Hybrid Model (YOLO detection with U-Net Segmentation)")
    st.write("3. YOLO Segmentation (Real-time Instance Segmentation Model)")

    st.divider()
    st.subheader("Select Your Model")

    model_choice = st.selectbox(
        "Choose Your Algorithm",
        [
            "Choose option",
            "U-Net",
            "Hybrid Model (YOLO Seg + U-Net)",
            "YOLO Segmentation"
        ]
    )

    if model_choice != "Choose option":

        uploaded_file = st.file_uploader(
            "Upload MRI Scan",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)

            st.image(image, width="stretch")
            st.success("MRI scan uploaded successfully")

            # =========================
            # YOLO SEGMENTATION
            # =========================
            if model_choice == "YOLO Segmentation":
                model = load_yolo_seg()

                results = model(image, conf=0.25, verbose=False)
                result = results[0]

                annotated = result.plot()

                st.subheader("Segmentation Result")
                st.image(annotated, width="stretch")

                st.subheader("Detected Objects")
                boxes = result.boxes.cpu().numpy()

                if len(boxes) > 0:
                    cols = st.columns(len(boxes))
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)

                        if x2 > x1 and y2 > y1:
                            crop = img_np[y1:y2, x1:x2]

                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = result.names[cls_id]

                            with cols[i]:
                                st.write(f"{class_name} ({conf:.2f})")
                                st.image(crop)
                else:
                    st.warning("No objects detected")

            else:
                st.info(f"Selected model: {model_choice}")
                st.info("This pipeline is not implemented yet")

    else:
        st.info("Please select a model to continue")
