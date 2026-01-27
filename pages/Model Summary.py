import streamlit as st

st.set_page_config(
    page_title="Model Summary",
    page_icon="🎈",
    layout="wide"
)

left, center, right = st.columns([1, 2, 1])

with center:
    st.title("Welcome!")

    st.markdown(
        "This is Brain Tumor Detection and Segmentation Senior Project "
        "Having 3 methods"
    )

    st.markdown("1. U-Net (Semantic Segmentation Model)")
    st.markdown("2. Hybrid Model (YOLO detection with U-Net Segmentation)")
    st.markdown("3. YOLO Segmentation (Real-time Instance Segmentation Model)")

    tab_accuracy, tab_dataset = st.tabs(["Accuracy", "Dataset"])

    with tab_accuracy:
        algorithm = st.selectbox(
            "Choose Your Algorithm",
            ["Choose option", "Pure U-Net", "Hybrid Model (YOLO + U-Net)", "Pure YOLO"]
        )

        if algorithm == "Pure U-Net":
            st.error("CNN Accuracy : 96.4%")

        elif algorithm == "Hybrid Model (YOLO + U-Net)":
            st.error("ANN Accuracy : 91.2%")

        elif algorithm == "Pure Yolo":
            st.error("SVM Accuracy : 88.7%")

    with tab_dataset:
        st.write("Dataset Information")
        st.write("Brain MRI Images")
        st.write("Classes:")
        st.write("- Glioma")
        st.write("- Meningioma")
        st.write("- Pituitary")
        st.write("- No Tumor")
        st.write("Image Size: 224 x 224")
        st.write("Split: Train / Test")