import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Input MRI Scan",
    page_icon="🎈",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CLASS NAMES + COLORS
# =========================
CLASS_NAMES = {
    0: "no tumor",
    1: "pituitary",
    2: "meningioma",
    3: "glioma"
}

COLORS = np.array([
    [0,   0,   0],
    [255, 0,   0],
    [0,   255, 0],
    [0,   0, 255],
], dtype=np.uint8)

# =========================
# LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_yolo_seg():
    from ultralytics import YOLO
    return YOLO("models/3cls/direct/best_yolo_direct_3cls.pt")

# =========================
# LOAD UNET MODEL (.pth)
# =========================
@st.cache_resource
def load_unet():
    from segmentation_models_pytorch import Unet
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=len(CLASS_NAMES)
    )
    model.load_state_dict(
        torch.load("models/3cls/best_unet_transfer_3cls.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_yolo_detect():
    from ultralytics import YOLO
    return YOLO("models/best_yolo_hybrid_transfer.pt")

@st.cache_resource
def load_hybrid_unet():
    from segmentation_models_pytorch import Unet
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(
        torch.load("models/best_unet_hybrid_transfer.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model
# =========================
# UTILS
# =========================
def preprocess_unet(img):
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.unsqueeze(0)

def mask_to_color(mask, cls):
    h, w = mask.shape
    color = COLORS[cls] 
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask != 0] = color
    return color_mask

def extract_boxes(mask):
    boxes = []
    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue

        binary = (mask == cls_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)

        for lbl in range(1, num_labels):
            ys, xs = np.where(labels == lbl)
            if len(xs) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

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

            if model_choice == "U-Net":
                import albumentations as A
                from albumentations.pytorch import ToTensorV2

                model = load_unet()
                model.eval()

                # =========================
                # DEFINE VAL TRANSFORM (INLINE)
                # =========================
                val_tf = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(),
                    ToTensorV2()
                ])

                # =========================
                # APPLY TRANSFORM (CM-CONSISTENT)
                # =========================
                aug = val_tf(image=img_np)
                img_tensor = aug["image"].unsqueeze(0).to(device)  # [1,3,512,512]

                with torch.no_grad():
                    logits = model(img_tensor)
                    pred_mask_raw = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

                # =========================
                # IMAGE-LEVEL CLASS (CM LOGIC)
                # =========================
                tumor_pixels = pred_mask_raw[pred_mask_raw != 0]

                if tumor_pixels.size == 0:
                    image_cls = 0
                    confidence = 0.0
                else:
                    values, counts = np.unique(tumor_pixels, return_counts=True)
                    image_cls = int(values[counts.argmax()])
                    confidence = float(tumor_pixels.size / pred_mask_raw.size)

                st.subheader("Predicted Class")
                st.success(f"{CLASS_NAMES[image_cls]})")

                # =========================
                # RESIZE MASK FOR DISPLAY
                # =========================
                h, w, _ = img_np.shape
                pred_mask_vis = cv2.resize(
                    pred_mask_raw,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

                # =========================
                # CREATE COLOR MASK (CLASS COLOR)
                # =========================
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                color_mask[pred_mask_vis != 0] = COLORS[image_cls]

                # =========================
                # OVERLAY
                # =========================
                overlay = (0.6 * img_np + 0.4 * color_mask).astype(np.uint8)

                # =========================
                # COMPUTE BOUNDING BOX
                # =========================
                ys, xs = np.where(pred_mask_vis != 0)

                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()

                    box_color = tuple(int(c) for c in COLORS[image_cls])

                    # draw bounding box
                    cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        thickness=2
                    )

                    label = f"{CLASS_NAMES[image_cls]}"

                    # label background
                    (tw, th), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        2
                    )

                    cv2.rectangle(
                        overlay,
                        (x1, y1 - th - 8),
                        (x1 + tw + 6, y1),
                        box_color,
                        -1
                    )

                    # label text
                    cv2.putText(
                        overlay,
                        label,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # =========================
                # DISPLAY RESULT
                # =========================
                st.subheader("Segmentation Result")
                st.image(overlay, width="stretch")

            # =========================
            # YOLO SEGMENTATION
            # =========================
            elif model_choice == "YOLO Segmentation":
                model = load_yolo_seg()

                results = model(image, conf=0.3, verbose=False)
                result = results[0]

                annotated = result.plot(conf=False)

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
                            class_name = result.names[cls_id]

                            with cols[i]:
                                st.write(f"{class_name}")
                                st.image(crop)
                else:
                    st.warning("No objects detected")

            elif model_choice == "Hybrid Model (YOLO Seg + U-Net)":

                yolo = load_yolo_detect()
                unet = load_hybrid_unet()
                unet.eval()

                # =========================
                # YOLO DETECTION + CLASS VOTE
                # =========================
                results = yolo(image, conf=0.3, verbose=False)
                result = results[0]

                boxes = result.boxes

                # ---- YOLO CLASS VOTE ----
                if boxes is not None and len(boxes) > 0:
                    yolo_cls = boxes.cls.cpu().numpy().astype(int)
                    values, counts = np.unique(yolo_cls, return_counts=True)
                    voted_cls = int(values[counts.argmax()])

                    # YOLO → segmentation class mapping
                    yolo_map = {0: 3, 1: 1, 2: 2}
                    image_cls = yolo_map.get(voted_cls, 0)
                else:
                    image_cls = 0  # no tumor

                # =========================
                # HYBRID SEGMENTATION (BINARY)
                # =========================
                h, w, _ = img_np.shape
                pred_mask = np.zeros((h, w), dtype=np.float32)

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = img_np[y1:y2, x1:x2]

                    # preprocess (same logic as offline)
                    crop_resized = cv2.resize(crop, (512, 512))
                    crop_tensor = (
                        torch.from_numpy(crop_resized / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )

                    with torch.no_grad():
                        seg = torch.sigmoid(unet(crop_tensor))[0, 0].cpu().numpy()

                    seg = cv2.resize(
                        seg,
                        (x2 - x1, y2 - y1),
                        interpolation=cv2.INTER_NEAREST
                    )

                    pred_mask[y1:y2, x1:x2] = np.maximum(
                        pred_mask[y1:y2, x1:x2],
                        seg
                    )

                hybrid_mask = (pred_mask > 0).astype(np.uint8)

                # =========================
                # COLOR OVERLAY (YOLO CLASS)
                # =========================
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                color_mask[hybrid_mask != 0] = COLORS[image_cls]

                overlay = (0.6 * img_np + 0.4 * color_mask).astype(np.uint8)

                # =========================
                # MERGED BOUNDING BOX
                # =========================
                ys, xs = np.where(hybrid_mask != 0)

                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()

                    box_color = tuple(int(c) for c in COLORS[image_cls])

                    cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    label = CLASS_NAMES[image_cls]

                    (tw, th), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        2
                    )

                    cv2.rectangle(
                        overlay,
                        (x1, y1 - th - 8),
                        (x1 + tw + 6, y1),
                        box_color,
                        -1
                    )

                    cv2.putText(
                        overlay,
                        label,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # =========================
                # DISPLAY
                # =========================
                st.subheader("Predicted Class")
                st.success(CLASS_NAMES[image_cls])

                st.subheader("Hybrid Segmentation Result")
                st.image(overlay, width="stretch")
    else:
        st.info("Please select a model to continue")
