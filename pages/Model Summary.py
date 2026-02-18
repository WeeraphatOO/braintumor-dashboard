import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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

    tab_accuracy, tab_architecture, tab_dataset = st.tabs(["Accuracy", "Model Architecture", "Dataset"])

    with tab_accuracy:
        algorithm = st.selectbox(
            "Choose Your Algorithm",
            ["Pure U-Net", "Hybrid Model (YOLO + U-Net)", "Pure YOLO"]
        )

        if algorithm == "Pure U-Net":
            st.subheader("U-Net Performance Comparison")
            st.info("Direct Training vs Transfer Learning")
            TRANSFER_CM = np.array([
                [242,   1,   5,   8],
                [  1, 293,   2,   4],
                [  3,   3, 297,   4],
                [  0,   0,   0, 140]
            ])

            DIRECT_CM = np.array([
                [228,   2,   4,  22],
                [  1, 296,   2,   2],
                [  8,   4, 298,   3],
                [  0,   0,   0, 140]
            ])

            TRANSFER_DICE = [0.7969, 0.86285, 0.94147]
            TRANSFER_IOU  = [0.66237, 0.75879, 0.88941]

            DIRECT_DICE = [0.77029, 0.8625, 0.92275]
            DIRECT_IOU  = [0.6264, 0.75824, 0.85659]

            # =========================
            # CSV PATHS
            # =========================
            DIRECT_CSV_PATH = BASE_DIR / "utils" / "train_history_unet_direct_3cls.csv"
            TRANSFER_CSV_PATH = BASE_DIR / "utils" / "train_history_unet_transfer_3cls.csv"

            col1, col2 = st.columns(2)

            # =========================
            # DIRECT U-Net LOSS
            # =========================
            with col1:
                st.markdown("### Direct U-Net Loss")
                df = pd.read_csv(DIRECT_CSV_PATH)

                plt.figure()
                plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Total Loss")
                plt.title("Direct Training U-Net Total Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

            # =========================
            # TRANSFER U-Net LOSS
            # =========================
            with col2:
                st.markdown("### Transfer Learning U-Net Loss")
                df = pd.read_csv(TRANSFER_CSV_PATH)

                plt.figure()
                plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Total Loss")
                plt.title("Direct Training U-Net Total Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()


            st.divider()

            # =========================
            # CONFUSION MATRICES
            # =========================
            CLASS_NAMES = ["glioma", "pituitary", "meningioma", "no tumor"]

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("### Direct U-Net Confusion Matrix")
                plt.figure(facecolor="black")
                plt.imshow(DIRECT_CM, cmap="cool")
                plt.title("Direct U-Net", color="white")
                plt.xlabel("Predicted", color="white")
                plt.ylabel("True", color="white")
                plt.xticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    rotation=45,
                    ha="right",
                    color="white"
                )
                plt.yticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    color="white"
                )

                for i in range(DIRECT_CM.shape[0]):
                    for j in range(DIRECT_CM.shape[1]):
                        plt.text(
                            j,
                            i,
                            DIRECT_CM[i, j],
                            ha="center",
                            va="center",
                            color="black"
                        )

                cbar = plt.colorbar()
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")

                st.pyplot(plt.gcf())
                plt.close()

            with col4:
                st.markdown("### Transfer U-Net Confusion Matrix")
                plt.figure(facecolor="black")
                plt.imshow(TRANSFER_CM, cmap="cool")
                plt.title("Transfer U-Net", color="white")
                plt.xlabel("Predicted", color="white")
                plt.ylabel("True", color="white")
                plt.xticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    rotation=45,
                    ha="right",
                    color="white"
                )
                plt.yticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    color="white"
                )

                for i in range(TRANSFER_CM.shape[0]):
                    for j in range(TRANSFER_CM.shape[1]):
                        plt.text(
                            j,
                            i,
                            TRANSFER_CM[i, j],
                            ha="center",
                            va="center",
                            color="black"
                        )

                cbar = plt.colorbar()
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")

                st.pyplot(plt.gcf())
                plt.close()

            st.divider()

            metrics_df = pd.DataFrame({
                "Class": ["glioma", "pituitary", "meningioma"],
                "Direct Dice": DIRECT_DICE,
                "Transfer Dice": TRANSFER_DICE,
                "Direct IoU": DIRECT_IOU,
                "Transfer IoU": TRANSFER_IOU
            })

            st.markdown("### Dice & IoU per Class")
            st.dataframe(metrics_df, width="stretch")

            st.divider()

            avg_metrics_df = pd.DataFrame({
                "Metric": ["Average Dice", "Average IoU"],
                "Direct U-Net": [
                    np.mean(DIRECT_DICE),
                    np.mean(DIRECT_IOU)
                ],
                "Transfer U-Net": [
                    np.mean(TRANSFER_DICE),
                    np.mean(TRANSFER_IOU)
                ]
            })

            st.markdown("### Average Dice & IoU Comparison")
            st.dataframe(avg_metrics_df, width="stretch")
            st.success(
                "Transfer Learning U-Net provides class-dependent results, with improvement observed mainly in glioma segmentation."
            )

        elif algorithm == "Hybrid Model (YOLO + U-Net)":

            st.subheader("Hybrid Model Performance")
            st.info(
                "Hybrid pipeline using YOLO for detection & classification "
                "and U-Net for tumor segmentation (trained separately)."
            )

            st.info(
                "Hybrid pipeline using YOLO for tumor detection & classification "
                "and U-Net for fine-grained tumor segmentation. "
                "Both models are trained separately and combined at inference time."
            )

            st.warning(
                "Note: The confusion matrix shown below is identical to Pure YOLO. "
                "In the Hybrid model, YOLO is responsible for detection and classification, "
                "while U-Net performs segmentation only."
            )

            # =========================
            # REUSE YOLO CONFUSION MATRIX
            # =========================
            CLASS_NAMES = ["glioma", "pituitary", "meningioma", "no tumor"]

            TRANSFER_CM = np.array([
                [242, 1, 5, 8],
                [1, 293, 2, 4],
                [3, 3, 297, 4],
                [0, 0, 0, 140]
            ])

            DIRECT_CM = np.array([
                [228, 2, 4, 22],
                [1, 296, 2, 2],
                [8, 4, 298, 3],
                [0, 0, 0, 140]
            ])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Direct YOLO Confusion Matrix (Hybrid Detection)")
                plt.figure(facecolor="black")
                plt.imshow(DIRECT_CM, cmap="cool")
                plt.title("Hybrid Model (YOLO Detect)", color="white")
                plt.xlabel("Predicted", color="white")
                plt.ylabel("True", color="white")
                plt.xticks(range(4), CLASS_NAMES, rotation=45, ha="right", color="white")
                plt.yticks(range(4), CLASS_NAMES, color="white")

                for i in range(4):
                    for j in range(4):
                        plt.text(j, i, DIRECT_CM[i, j], ha="center", va="center", color="black")

                cbar = plt.colorbar()
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")

                st.pyplot(plt.gcf())
                plt.close()

            with col2:
                st.markdown("### Transfer YOLO Confusion Matrix (Hybrid Detection)")
                plt.figure(facecolor="black")
                plt.imshow(TRANSFER_CM, cmap="cool")
                plt.title("Hybrid Model (YOLO Detect)", color="white")
                plt.xlabel("Predicted", color="white")
                plt.ylabel("True", color="white")
                plt.xticks(range(4), CLASS_NAMES, rotation=45, ha="right", color="white")
                plt.yticks(range(4), CLASS_NAMES, color="white")

                for i in range(4):
                    for j in range(4):
                        plt.text(j, i, TRANSFER_CM[i, j], ha="center", va="center", color="black")

                cbar = plt.colorbar()
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")

                st.pyplot(plt.gcf())
                plt.close()


            st.success(
                "In the Hybrid model, YOLO detection is equivalent to YOLO segmentation "
                "without using the segmentation head. "
                "Therefore, the classification performance remains identical to Pure YOLO, "
                "while U-Net improves spatial segmentation quality."
            )
            st.divider()

            # =========================
            # CSV PATHS (REUSE EXISTING)
            # =========================
            UNET_DIRECT_CSV = BASE_DIR / "utils" / "yolo_crop_unet_train_history_direct_3cls.csv"
            UNET_TRANSFER_CSV = BASE_DIR / "utils" / "yolo_crop_unet_train_history_transfer.csv"

            YOLO_DIRECT_CSV = BASE_DIR / "utils" / "yolo_for_unet_transfer.csv"
            YOLO_TRANSFER_CSV = BASE_DIR / "utils" / "yolo_for_unet_direct.csv"

            # =========================
            # U-NET LOSS (SEGMENTATION)
            # =========================
            st.markdown("## U-Net Segmentation Loss")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("### Direct U-Net")
                df = pd.read_csv(UNET_DIRECT_CSV)

                plt.figure()
                plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Direct U-Net Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

            with col4:
                st.markdown("### Transfer Learning U-Net")
                df = pd.read_csv(UNET_TRANSFER_CSV)

                plt.figure()
                plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Transfer U-Net Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

            st.divider()

            # =========================
            # YOLO LOSS (DETECTION + SEG)
            # =========================
            st.markdown("## YOLO Detection Loss")

            loss_cols = [
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss"
            ]

            val_loss_cols = [
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss"
            ]

            col5, col6 = st.columns(2)

            with col5:
                st.markdown("### Direct YOLO")
                df = pd.read_csv(YOLO_DIRECT_CSV)

                df["train_total_loss"] = df[loss_cols].sum(axis=1)
                df["val_total_loss"] = df[val_loss_cols].sum(axis=1)

                plt.figure()
                plt.plot(df["epoch"], df["train_total_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_total_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Direct YOLO Total Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

            with col6:
                st.markdown("### Transfer Learning YOLO")
                df = pd.read_csv(YOLO_TRANSFER_CSV)

                df["train_total_loss"] = df[loss_cols].sum(axis=1)
                df["val_total_loss"] = df[val_loss_cols].sum(axis=1)

                plt.figure()
                plt.plot(df["epoch"], df["train_total_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_total_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Transfer YOLO Total Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()


            # =========================
            # HYBRID EXPLANATION
            # =========================
            st.success(
                "The Hybrid model does not introduce a new training loss. "
                "YOLO and U-Net are trained independently and combined at inference time, "
                "allowing YOLO to guide localization while U-Net performs fine-grained segmentation."
            )
            st.divider()
            direct_iou = [0.8077, 0.9067, 0.6733]
            direct_dice = [0.8936, 0.9510, 0.8047]

            transfer_iou = [0.7789, 0.8989, 0.6549]
            transfer_dice = [0.8757, 0.9468, 0.7915]

            st.markdown("### Dice & IoU (Hybrid Model)")

            classes = ["glioma", "pituitary", "meningioma"]

            hybrid_metrics_df = pd.DataFrame({
                "Class": classes,

                # Direct
                "Direct Dice (Hybrid)": direct_dice,
                "Direct IoU (Hybrid)": direct_iou,

                # Transfer
                "Transfer Dice (Hybrid)": transfer_dice,
                "Transfer IoU (Hybrid)": transfer_iou,
            })

            st.dataframe(hybrid_metrics_df, width="stretch")

            st.divider()
            st.markdown("### Average Dice & IoU (Hybrid Model)")

            avg_hybrid_df = pd.DataFrame({
                "Metric": ["Average Dice", "Average IoU"],
                "Direct Hybrid": [
                    np.mean(direct_dice),
                    np.mean(direct_iou),
                ],
                "Transfer Hybrid": [
                    np.mean(transfer_dice),
                    np.mean(transfer_iou),
                ]
            })

            st.dataframe(avg_hybrid_df, width="stretch")
            st.success(
                "Compared to direct training, the transfer-based Hybrid model shows slightly lower Dice and IoU scores across all tumor classes. "
                "However, the overall segmentation performance remains stable, with pituitary tumors consistently achieving the highest accuracy."
            )




        elif algorithm == "Pure YOLO":
            st.subheader("YOLO Performance Comparison")
            st.info("Direct Training vs Transfer Learning")
            TRANSFER_CM = np.array([
                [242, 1, 5, 8],
                [1, 293, 2, 4],
                [3, 3, 297, 4],
                [0, 0, 0, 140]
            ])

            DIRECT_CM = np.array([
                [228, 2, 4, 22],
                [1, 296, 2, 2],
                [8, 4, 298, 3],
                [0, 0, 0, 140]
            ])

            TRANSFER_DICE = [0.7969, 0.86285, 0.94147]
            TRANSFER_IOU  = [0.66237, 0.75879, 0.88941]

            DIRECT_DICE = [0.77029, 0.8625, 0.92275]
            DIRECT_IOU  = [0.6264, 0.75824, 0.85659]

            # =========================
            # CSV PATHS
            # =========================
            DIRECT_CSV_PATH = BASE_DIR / "utils" / "direct_yolo_3cls_results.csv"
            TRANSFER_CSV_PATH = BASE_DIR / "utils" / "merged_yolo_transfer_3cls_results.csv"

            loss_cols = [
                "train/box_loss",
                "train/seg_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "train/sem_loss"
            ]

            val_loss_cols = [
                "val/box_loss",
                "val/seg_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "val/sem_loss"
            ]

            col1, col2 = st.columns(2)

            # =========================
            # DIRECT YOLO LOSS
            # =========================
            with col1:
                st.markdown("### Direct YOLO Loss")

                df = pd.read_csv(DIRECT_CSV_PATH)

                df["train_total_loss"] = df[loss_cols].sum(axis=1)
                df["val_total_loss"] = df[val_loss_cols].sum(axis=1)
                
                plt.figure()
                plt.plot(df["epoch"], df["train_total_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_total_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Total Loss")
                plt.title("Direct Training YOLO Total Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

            # =========================
            # TRANSFER YOLO LOSS
            # =========================
            with col2:
                st.markdown("### Transfer Learning YOLO Loss")

                df = pd.read_csv(TRANSFER_CSV_PATH)
                df["train_total_loss"] = df[loss_cols].sum(axis=1)
                df["val_total_loss"] = df[val_loss_cols].sum(axis=1)

                plt.figure()
                plt.plot(df["epoch"], df["train_total_loss"], label="Train Loss")
                plt.plot(df["epoch"], df["val_total_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Total Loss")
                plt.title("Transfer YOLO Total Loss")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

            st.divider()

            # =========================
            # CONFUSION MATRICES
            # =========================
            CLASS_NAMES = ["glioma", "pituitary", "meningioma", "no tumor"]

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("### Direct YOLO Confusion Matrix")
                plt.figure(facecolor="black")
                plt.imshow(DIRECT_CM, cmap="cool")
                plt.title("Direct YOLO", color="white")
                plt.xlabel("Predicted", color="white")
                plt.ylabel("True", color="white")
                plt.xticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    rotation=45,
                    ha="right",
                    color="white"
                )
                plt.yticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    color="white"
                )

                for i in range(DIRECT_CM.shape[0]):
                    for j in range(DIRECT_CM.shape[1]):
                        plt.text(
                            j,
                            i,
                            DIRECT_CM[i, j],
                            ha="center",
                            va="center",
                            color="black"
                        )

                cbar = plt.colorbar()
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")

                st.pyplot(plt.gcf())
                plt.close()

            with col4:
                st.markdown("### Transfer YOLO Confusion Matrix")
                plt.figure(facecolor="black")
                plt.imshow(TRANSFER_CM, cmap="cool")
                plt.title("Transfer YOLO", color="white")
                plt.xlabel("Predicted", color="white")
                plt.ylabel("True", color="white")
                plt.xticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    rotation=45,
                    ha="right",
                    color="white"
                )
                plt.yticks(
                    range(len(CLASS_NAMES)),
                    CLASS_NAMES,
                    color="white"
                )

                for i in range(TRANSFER_CM.shape[0]):
                    for j in range(TRANSFER_CM.shape[1]):
                        plt.text(
                            j,
                            i,
                            TRANSFER_CM[i, j],
                            ha="center",
                            va="center",
                            color="black"
                        )

                cbar = plt.colorbar()
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")

                st.pyplot(plt.gcf())
                plt.close()

            st.divider()

            metrics_df = pd.DataFrame({
                "Class": ["glioma", "pituitary", "meningioma"],
                "Direct Dice": DIRECT_DICE,
                "Transfer Dice": TRANSFER_DICE,
                "Direct IoU": DIRECT_IOU,
                "Transfer IoU": TRANSFER_IOU
            })

            st.markdown("### Dice & IoU per Class")
            st.dataframe(metrics_df, width="stretch")

            st.divider()

            avg_metrics_df = pd.DataFrame({
                "Metric": ["Average Dice", "Average IoU"],
                "Direct YOLO": [
                    np.mean(DIRECT_DICE),
                    np.mean(DIRECT_IOU)
                ],
                "Transfer YOLO": [
                    np.mean(TRANSFER_DICE),
                    np.mean(TRANSFER_IOU)
                ]
            })

            st.markdown("### Average Dice & IoU Comparison")
            st.dataframe(avg_metrics_df, width="stretch")
            st.success("Transfer Learning YOLO achieves consistently better Dice and IoU across tumor classes")

    with tab_dataset:
        st.write("Dataset Information")
        st.write("Brain MRI Images")
        st.write("Classes:")
        st.write("- Glioma")
        st.write("- Meningioma")
        st.write("- Pituitary")
        st.write("- No Tumor")
        st.link_button('Dataset Link','https://www.kaggle.com/datasets/briscdataset/brisc2025/data')
    with tab_architecture:

        algorithm = st.selectbox(
            "Choose Your Algorithm",
            [
                "U-Net Architecture",
                "Hybrid Architecture",
                "YOLO Segmentation Architecture"
            ]
        )

        # =====================================================
        # U-NET ARCHITECTURE
        # =====================================================
        if algorithm == "U-Net Architecture":
            st.title("ResUNet Architecture")

            st.image(
                "assets/resunet.png",
                caption="ResUNet Architecture",
                width= "stretch"
            )

            st.markdown("""
            ResUNet is an improved version of U-Net that integrates residual learning 
            into the encoder–decoder segmentation framework.

            It preserves the classical U-shaped structure while replacing standard 
            convolution blocks with residual blocks to improve training stability 
            and gradient propagation.
            """)
            st.divider()
            st.markdown("### Encoder (Feature Extraction)")

            st.markdown("""
            The encoder gradually reduces spatial resolution while increasing 
            feature depth to extract high-level representations.

            Each encoder stage consists of:
            - Residual Block (Conv → BN → ReLU → Conv → BN + Skip Connection)
            - Downsampling (MaxPooling or Strided Convolution)

            Purpose:
            - Capture spatial features
            - Extract texture and tumor patterns
            - Learn hierarchical representations
            """)
            st.divider()
            st.markdown("### Bottleneck (Neck / Bridge Layer)")

            st.markdown("""
            The bottleneck is the deepest part of the network.

            It contains a residual block without downsampling.

            Purpose:
            - Capture global context
            - Combine spatial and semantic information
            - Learn abstract tumor representation
            """)
            st.divider()
            st.markdown("### Decoder (Reconstruction Path)")

            st.markdown("""
            The decoder gradually restores spatial resolution.

            Each decoder stage consists of:
            - Upsampling (Transpose Convolution or Bilinear Upsample)
            - Concatenation with corresponding encoder features (Skip Connection)
            - Residual Block

            Purpose:
            - Recover spatial resolution
            - Refine segmentation boundaries
            - Combine low-level and high-level features
            """)

            st.divider()
            st.markdown("### What is Skip Connections?")

            st.markdown("""
            Skip connections transfer feature maps from the encoder 
            directly to the decoder at the same spatial level.

            Purpose:
            - Preserve fine-grained details
            - Improve boundary accuracy
            - Prevent information loss during downsampling
            """)
            st.divider()


        # =====================================================
        # HYBRID ARCHITECTURE
        # =====================================================
        elif algorithm == "Hybrid Architecture":

            st.title("Hybrid Architecture: YOLO Detection + U-Net Segmentation")
            st.markdown("""
            This hybrid model combines fast object detection (YOLO) with precise segmentation (U-Net).
            The key idea is Region of Interest (ROI) cropping before segmentation.
            """)

            # =========================================================
            # STAGE 1 - YOLO DETECTION
            # =========================================================
            st.markdown("### Stage 1 - YOLO Detection")
            st.image(
                "assets/yolo_detect.png",
                caption="YOLO Detection Model Architecture",
                width= "stretch"
            )

            st.markdown("""
            The full input image is first passed into a YOLO detection model.
            Each predicted bounding box defines a potential Region of Interest (ROI).            
            """)

            st.markdown("""
            YOLO predicts:

            - Bounding box coordinates (x, y, w, h)
            - Object confidence
            - Class probability
            """)


            st.markdown("""
            Each predicted bounding box defines a potential Region of Interest (ROI).
            """)

            st.divider()

            # =========================================================
            # STAGE 2 - ROI CROP (CORE PROCESS)
            # =========================================================
            st.markdown("### Stage 2 - ROI Crop")

            st.markdown("""
            Instead of segmenting the whole image, we crop only detected regions.
            This reduces background noise and improves segmentation precision by making Unet focus only tumor area.
            """)

            st.image(
                "assets/crop_roi.png",
                caption="ROI Cropping Process",
                width= "stretch"
            )

            st.divider()

            # =========================================================
            # STAGE 3 - U-NET SEGMENTATION
            # =========================================================
            st.markdown("### Stage 3 - U-Net Segmentation")
            st.image(
                "assets/resunet.png",
                caption="ResUNet Architecture",
                width= "stretch"
            )
            st.markdown("""
            Each resized ROI is independently passed into U-Net.
            """)

            st.divider()

            # =========================================================
            # STAGE 4 - RESTORE TO ORIGINAL IMAGE
            # =========================================================
            st.markdown("### Stage 4 - Restore Mask to Original Image")

            st.markdown("""
            The predicted ROI mask must be mapped back to original image coordinates.
            """)

            st.markdown("""
            Repeat for all detected objects.
            """)

            st.markdown("""
            Final output contains:

            - Bounding boxes (from YOLO)
            - Segmentation masks (from U-Net)
            """)

            st.divider()

        # =====================================================
        # YOLO SEGMENTATION ARCHITECTURE
        # =====================================================
        elif algorithm == "YOLO Segmentation Architecture":
            st.title("YOLO Segmentation Architecture (YOLO-Seg)")
            st.image(
                "assets/yolo_segment.png",
                caption="YOLOv11 Segmentation Model Architecture",
                width= "stretch"
            )
            st.markdown("Model architecture explanation divided into Backbone, Neck, Detection Head, and Segmentation Head.")

            # =========================================================
            # BACKBONE
            # =========================================================
            st.markdown("### Backbone - Feature Extraction")

            st.markdown("""
            The backbone extracts hierarchical features from the 640×640 input image.
            It progressively reduces spatial resolution while increasing channel depth.
            """)

            st.subheader("CBS Block")
            st.markdown("""
            **Structure:** Conv → BatchNorm → SiLU  
            **Purpose:** Basic feature extraction block used throughout the network.
            """)

            st.subheader("C3K2 Block")
            st.markdown("""
            **Structure:** CBS → C3K (N times) → Concat → CBS  
            **Purpose:** Efficient deep feature extraction using CSP-style connections.
            """)

            st.subheader("C3K Block")
            st.markdown("""
            **Structure:** CBS → Bottleneck (N times) → Concat → CBS  
            **Purpose:** Enhances representation capacity while maintaining efficiency.
            """)

            st.subheader("Bottleneck")
            st.markdown("""
            **Structure:** CBS → CBS + Residual Shortcut  
            **Purpose:** Enables residual learning and stabilizes deep training.
            """)

            st.subheader("SPPF (Spatial Pyramid Pooling Fast)")
            st.markdown("""
            **Structure:** CBS → MaxPool (multiple times) → Concat → CBS  
            **Purpose:** Expands receptive field and captures multi-scale context.
            """)

            st.subheader("C2PSA (Cross Partial Self Attention)")
            st.markdown("""
            **Structure:** CBS → PSA (N times) → Concat → CBS  
            **Purpose:** Adds attention mechanism to focus on important regions.
            """)

            st.divider()

            # =========================================================
            # NECK
            # =========================================================
            st.markdown("### Neck - Multi-scale Feature Fusion")

            st.markdown("""
            The neck combines features from different scales using an FPN/PAN-like structure.
            It helps detect both small and large objects.
            """)

            st.subheader("Upsample")
            st.markdown("Increases spatial resolution to recover fine-grained details.")

            st.subheader("Concat")
            st.markdown("Merges shallow and deep features to improve multi-scale representation.")

            st.subheader("C3K2 + CBS in Neck")
            st.markdown("Used after feature concatenation to refine fused features.")

            st.divider()

            # =========================================================
            # DETECTION HEAD
            # =========================================================
            st.header("Detection Head - Bounding Box & Classification")

            st.markdown("""
            Three detection heads are used for multi-scale object prediction.
            Each head predicts bounding boxes and class probabilities.
            """)

            st.subheader("Box Branch")
            st.markdown("""
            **Structure:** CBS → CBS → Conv2D  
            **Output:** Bounding box coordinates (x, y, w, h)
            """)

            st.subheader("Classification Branch")
            st.markdown("""
            **Structure:** DWConv + CBS → DWConv + CBS → Conv2D  
            **Output:** Class probabilities
            """)

            st.subheader("Detection Output")
            st.markdown("Outputs from all scales are concatenated to produce final predictions.")

            st.divider()

            # =========================================================
            # SEGMENTATION HEAD
            # =========================================================
            st.header("Segmentation Head - Instance Mask Generation")

            st.markdown("""
            YOLO-Seg extends object detection by generating pixel-level instance masks.
            The segmentation head consists of Prototype Mask generation and Mask Coefficients.
            """)

            # ---------------------------------------------------------
            # PROTOTYPE MASK (DETAILED)
            # ---------------------------------------------------------
            st.subheader("Prototype Mask")

            st.markdown("""
            The Prototype Mask branch generates **K shared mask bases** (prototypes).

            Instead of predicting a full-resolution mask for each object, the model:

            1. Generates a fixed set of global mask prototypes.
            2. Combines them differently for each detected object.
            """)

            st.markdown("### Prototype Masks Benefit?")

            st.markdown("""
            Directly predicting one mask per object would be computationally expensive.

            Prototype-based design:

            - Reduces memory usage  
            - Speeds up inference  
            - Enables real-time segmentation  
            - Shares spatial information across objects  
            """)

            st.divider()

            # ---------------------------------------------------------
            # MASK COEFFICIENT
            # ---------------------------------------------------------
            st.subheader("Mask Coefficient")

            st.markdown("""
            For each detected object, the detection head predicts K coefficients:

            Coefficient shape per object:
            """)

            st.latex(r"""
            c_i = (c_{i1}, c_{i2}, ..., c_{iK})
            """)

            st.markdown("""
            These coefficients determine how much each prototype contributes to the final mask.
            """)

            st.divider()

            # ---------------------------------------------------------
            # MASK CALCULATION
            # ---------------------------------------------------------
            st.subheader("Mask Calculation")

            st.latex(r"""
            M_i(x, y) = \sigma \left( \sum_{k=1}^{K} c_{ik} \cdot P_k(x, y) \right)
            """)

            st.markdown(r"""
            Where:

            - $M_i(x, y)$ = final mask for object $i$  
            - $c_{ik}$ = mask coefficient  
            - $P_k(x, y)$ = prototype mask  
            - $\sigma$ = sigmoid activation  
            """)

            st.markdown("""
            After this step:

            - The mask is resized to the original image size.
            - It is cropped using the predicted bounding box.
            - Thresholding is applied to obtain the binary segmentation mask.
            """)

            st.divider()
