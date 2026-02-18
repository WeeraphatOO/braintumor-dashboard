import streamlit as st

st.set_page_config(
    page_title="Dashboard Home",
    page_icon="🏠",
    layout="wide"
)

st.title("Dashboard Home")

st.markdown("""
### Brain Tumor Detection and Segmentation System Using CNNs

Welcome to the **Brain Tumor Detection and Segmentation** dashboard.

This project focuses on applying **CNNs techniques** to analyze brain MRI images for:
- **Tumor detection** (localizing tumor regions)
- **Tumor segmentation** (pixel-level classification of tumor areas)
- **Multi-class tumor analysis** across different tumor types

#### Project Purpose
The main objectives of this project are:
- To develop and compare **multiple CNN models** for brain tumor analysis  
- To evaluate the performance of **segmentation-based and detection-based approaches**
- To provide clear visualizations and metrics for model evaluation

#### Models Implemented
- **U-Net–based segmentation models**
- **YOLO segmentation models**
- **Hybrid models (YOLO + U-Net)** for region-focused segmentation

#### What You Can Explore
Using the sidebar, you can navigate to:
- Model performance metrics (e.g., Dice score, IoU)
- Dataset distribution and class analysis
- Experimental results and observations
- Testing Your MRI Scans

This dashboard is designed to support **analysis, comparison, and interpretation** of different approaches in brain tumor detection and segmentation.
""")
st.divider()
st.markdown("""
Developed By
Weeraphat Oun-Ob  
วีรภัทร์ อุ่นอบ
Bachelor of Computer Engineering  

Project Advisor
Asst. Prof. Dr. Thanawin Rakthanmanon 
ผู้ช่วยศาสตราจารย์ ดร.ธนาวินท์ รักธรรมานนท์
Department of Computer Engineering  
---
""")