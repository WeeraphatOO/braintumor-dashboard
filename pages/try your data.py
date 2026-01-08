import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Input MRI Scan",
    page_icon="🎈",
    layout="wide"
)

# =========================
# CENTER LAYOUT
# =========================
left, center, right = st.columns([1, 2, 1])

with center:
    st.title("Input Your MRI Scan!")

    st.write(
        "You can upload a brain MRI scan and select a segmentation model "
        "to perform tumor detection and segmentation."
    )

    st.write("1. U-Net (Semantic Segmentation Model)")
    st.write("2. Hybrid Model (YOLO detection with U-Net Segmentation)")
    st.write("3. YOLO Segmentation (Real-time Instance Segmentation Model)")

    st.divider()

    st.subheader("Select Your Model")

    model = st.selectbox(
        "Choose Your Algorithm",
        [
            "Choose option",
            "U-Net",
            "Hybrid Model (YOLO Seg + U-Net)",
            "YOLO Segmentation"
        ]
    )

    # =========================
    # FILE UPLOADER
    # =========================
    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, width = 'stretch')
        st.success("MRI scan uploaded successfully")

        if model != "Choose option":
            st.info(f"Selected model: {model}")
            st.info("Prediction pipeline will be executed here")
        else:
            st.warning("Please select a model before prediction")
