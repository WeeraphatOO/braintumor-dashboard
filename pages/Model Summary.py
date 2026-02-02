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

    tab_accuracy, tab_dataset = st.tabs(["Accuracy", "Dataset"])

    with tab_accuracy:
        algorithm = st.selectbox(
            "Choose Your Algorithm",
            ["Choose option", "Pure U-Net", "Hybrid Model (YOLO + U-Net)", "Pure YOLO"]
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

            st.divider()

            st.success(
                "In the Hybrid model, YOLO detection is equivalent to YOLO segmentation "
                "without using the segmentation head. "
                "Therefore, the classification performance remains identical to Pure YOLO, "
                "while U-Net improves spatial segmentation quality."
            )

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

            st.divider()

            # =========================
            # HYBRID EXPLANATION
            # =========================
            st.success(
                "The Hybrid model does not introduce a new training loss. "
                "YOLO and U-Net are trained independently and combined at inference time, "
                "allowing YOLO to guide localization while U-Net performs fine-grained segmentation."
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
        st.write("Image Size: 224 x 224")
        st.write("Split: Train / Test")