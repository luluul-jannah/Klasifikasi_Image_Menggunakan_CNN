
import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. KONFIGURASI DASHBOARD
# ==========================================

st.set_page_config(
    page_title="OceanicLife CNN Dashboard",
    page_icon="🌊",
    layout="wide"
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(BASE_DIR, "output_oceaniclife")

MODEL_PATH = os.path.join(
    OUT_DIR,
    "final_marine_cnn_model.keras"
)
st.write("Lokasi aplikasi:", BASE_DIR)
st.write("Lokasi folder output:", OUT_DIR)
st.write("Lokasi model:", MODEL_PATH)
st.write("Model ditemukan:", os.path.exists(MODEL_PATH))
print("Lokasi model:", os.path.abspath(MODEL_PATH))
print("File model ditemukan:", os.path.exists(MODEL_PATH))

METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
HISTORY_PATH = os.path.join(OUT_DIR, "history.json")
EDA_PATH = os.path.join(OUT_DIR, "eda.json")
ANALYSIS_PATH = os.path.join(OUT_DIR, "analysis.json")
CM_PATH = os.path.join(OUT_DIR, "confusion_matrix.npy")
CLASS_REPORT_PATH = os.path.join(OUT_DIR, "per_class.csv")

IMG_SIZE = (160, 160)

# ==========================================
# 2. MEMUAT MODEL DAN DATA
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Periksa file yang dibutuhkan
if not os.path.exists(MODEL_PATH):
    st.error("Model tidak ditemukan! Periksa folder output_oceaniclife.")
    st.stop()

model = load_model()

metrics = load_json(METRICS_PATH)
history = load_json(HISTORY_PATH)
eda = load_json(EDA_PATH)
analysis = load_json(ANALYSIS_PATH)

class_names = metrics.get("class_names", [
    "Fish",
    "Coral",
    "Shark",
    "Octopuses",
    "Crustaceans",
    "Seaweed",
    "Sponges",
    "Other marine organisms"
])

# ==========================================
# 3. JUDUL DASHBOARD
# ==========================================

st.title("🌊 OceanicLife CNN Dashboard")

st.markdown("""
Dashboard klasifikasi organisme laut menggunakan metode
**Convolutional Neural Network (CNN)** dengan arsitektur
**MobileNetV2 Transfer Learning**.
""")

st.divider()

# ==========================================
# 4. MENU SIDEBAR
# ==========================================

menu = st.sidebar.radio(
    "Menu Dashboard",
    [
        "Beranda",
        "Informasi Dataset",
        "Evaluasi Model",
        "Prediksi Gambar"
    ]
)

# ==========================================
# 5. BERANDA
# ==========================================

if menu == "Beranda":

    st.header("Dashboard Klasifikasi OceanicLife")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Data Training",
        metrics.get("n_train", 0)
    )

    col2.metric(
        "Data Validation",
        metrics.get("n_val", 0)
    )

    col3.metric(
        "Data Testing",
        metrics.get("n_test", 0)
    )

    col4.metric(
        "Jumlah Kelas",
        len(class_names)
    )

    st.subheader("Informasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Arsitektur Model**")
        st.write("MobileNetV2")

    with col2:
        st.info("**Metode Pelatihan**")
        st.write("Transfer Learning dan Fine-Tuning")

    st.subheader("Kelas Organisme Laut")

    for i, name in enumerate(class_names, start=1):
        st.write(f"{i}. {name}")

# ==========================================
# 6. INFORMASI DATASET
# ==========================================

elif menu == "Informasi Dataset":

    st.header("Informasi Dataset")

    st.write(
        "Dataset OceanicLife digunakan untuk "
        "mengklasifikasikan gambar organisme laut "
        "ke dalam delapan kelas."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Training", metrics.get("n_train", 0))
    col2.metric("Validation", metrics.get("n_val", 0))
    col3.metric("Testing", metrics.get("n_test", 0))

    st.subheader("Daftar Kelas")

    class_table = pd.DataFrame({
        "No": range(1, len(class_names) + 1),
        "Nama Kelas": class_names
    })

    st.dataframe(class_table, use_container_width=True)

    if os.path.exists(CLASS_REPORT_PATH):
        st.subheader("Distribusi Data Testing")

        report = pd.read_csv(CLASS_REPORT_PATH)

        if "support" in report.columns:
            distribution = report[
                report["support"].notna()
            ][["support"]].copy()

            distribution.index.name = "Kelas"
            distribution = distribution[
                distribution["support"] > 0
            ]

            st.bar_chart(distribution["support"])

            st.caption(
                "Grafik menunjukkan jumlah data testing "
                "pada setiap kelas, bukan keseluruhan dataset."
            )

# ==========================================
# 7. EVALUASI MODEL
# ==========================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Kinerja Model CNN")

    macro = metrics.get("macro", {})
    weighted = metrics.get("weighted", {})

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0) * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{macro.get('precision', 0) * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{macro.get('recall', 0) * 100:.2f}%"
    )

    col4.metric(
        "F1-Score",
        f"{macro.get('f1', 0) * 100:.2f}%"
    )

    st.caption(
        "Precision, Recall, dan F1-Score di atas menggunakan "
        "nilai Macro Average."
    )

    # Classification report
    if os.path.exists(CLASS_REPORT_PATH):

        st.subheader("Classification Report")

        report = pd.read_csv(
            CLASS_REPORT_PATH,
            index_col=0
        )

        st.dataframe(
            report.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}"
            }),
            use_container_width=True
        )

    # Grafik training
    st.subheader("Grafik Accuracy dan Loss")

    if "accuracy" in history and "loss" in history:

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(
            history["accuracy"],
            label="Training Accuracy"
        )

        axes[0].plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        axes[0].set_title("Grafik Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(
            history["loss"],
            label="Training Loss"
        )

        axes[1].plot(
            history["val_loss"],
            label="Validation Loss"
        )

        axes[1].set_title("Grafik Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Confusion matrix
    st.subheader("Confusion Matrix")

    if os.path.exists(CM_PATH):

        cm = np.load(CM_PATH)

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Kelas Sebenarnya")
        ax.set_title("Confusion Matrix")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # Analisis
    st.subheader("Analisis Hasil Evaluasi")

    if analysis:

        for key, value in analysis.items():
            st.write(f"**{key.replace('_', ' ').title()}**")
            st.write(value)

# ==========================================
# 8. PREDIKSI GAMBAR
# ==========================================

elif menu == "Prediksi Gambar":

    st.header("Prediksi Organisme Laut")

    st.write(
        "Unggah gambar organisme laut untuk mengetahui "
        "hasil klasifikasi menggunakan model CNN."
    )

    uploaded_file = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Gambar yang Diunggah")

        st.image(
            image,
            caption="Gambar Input",
            width=350
        )

        if st.button("Mulai Prediksi", type="primary"):

            with st.spinner("Model sedang melakukan prediksi..."):

                # Resize gambar
                img = image.resize(IMG_SIZE)

                # Ubah menjadi array
                img_array = np.array(img).astype("float32")

                # Tambahkan dimensi batch
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Prediksi
                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                probabilities = prediction[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = (
                    probabilities[predicted_index] * 100
                )

                st.success("Prediksi berhasil!")

                st.subheader("Hasil Klasifikasi")

                st.markdown(
                    f"### Organisme: {predicted_class}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    float(probabilities[predicted_index])
                )

                # Probabilitas semua kelas
                st.subheader("Probabilitas Setiap Kelas")

                probability_df = pd.DataFrame({
                    "Kelas": class_names,
                    "Probabilitas": probabilities * 100
                })

                probability_df = probability_df.sort_values(
                    by="Probabilitas",
                    ascending=False
                )

                st.bar_chart(
                    probability_df.set_index("Kelas")
                )

                st.dataframe(
                    probability_df.style.format({
                        "Probabilitas": "{:.2f}%"
                    }),
                    use_container_width=True
                )

# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "OceanicLife CNN Dashboard | Klasifikasi Organisme Laut "
    "Menggunakan MobileNetV2"
)