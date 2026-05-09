import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import tempfile
import os
from datetime import datetime

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/heart_model.h5")

model = load_model()
CLASSES = ["Angina", "Cardio Vascular", "Coronary Artery", "Hypotension"]

# --- Simple Heatmap (Grad-CAM शिवाय) ---
def get_heatmap(img_array):
    img = np.array(img_array[0] * 255, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return heatmap, overlay

# --- PDF Report ---
def generate_pdf(patient_name, result, confidence, prediction, orig_img, heatmap_img, overlay_img):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFillColor(colors.HexColor("#1a1a2e"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 45, "AI Based Heart Disease Detection Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 65, f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 110, "Patient Information")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 130, f"Patient Name: {patient_name}")
    c.drawString(50, height - 148, f"Date: {datetime.now().strftime('%d-%m-%Y')}")

    c.setFillColor(colors.HexColor("#e8f5e9"))
    c.rect(45, height - 210, width - 90, 50, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#1b5e20"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(55, height - 178, f"Detected Disease: {result}")
    c.drawString(55, height - 198, f"Confidence: {confidence:.2f}%")

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 230, "Prediction Probabilities:")
    c.setFont("Helvetica", 11)
    y = height - 250
    for i, cls in enumerate(CLASSES):
        prob = prediction[0][i] * 100
        c.drawString(60, y, f"{cls}: {prob:.2f}%")
        c.setFillColor(colors.HexColor("#42a5f5"))
        c.rect(220, y - 2, int(prob * 2), 10, fill=True, stroke=False)
        c.setFillColor(colors.black)
        y -= 20

    def save_temp(img_array):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        Image.fromarray(np.uint8(img_array)).save(tmp.name)
        return tmp.name

    orig_path = save_temp(np.array(orig_img.resize((224, 224))))
    heat_path = save_temp(heatmap_img)
    over_path = save_temp(overlay_img)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 360, "Image Analysis:")
    img_y = height - 530
    c.drawImage(orig_path, 45, img_y, width=150, height=150)
    c.setFont("Helvetica", 10)
    c.drawString(85, img_y - 15, "Original")
    c.drawImage(heat_path, 215, img_y, width=150, height=150)
    c.drawString(255, img_y - 15, "Heatmap")
    c.drawImage(over_path, 385, img_y, width=150, height=150)
    c.drawString(425, img_y - 15, "Overlay")

    c.setFillColor(colors.HexColor("#1a1a2e"))
    c.rect(0, 0, width, 35, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 9)
    c.drawString(50, 12, "AI Based Heart Disease Detection | 2D Echo Image Analysis")

    c.save()
    buffer.seek(0)
    os.unlink(orig_path)
    os.unlink(heat_path)
    os.unlink(over_path)
    return buffer

# --- UI ---
st.set_page_config(page_title="Heart Disease Detection", page_icon="🫀", layout="wide")
st.title("🫀 AI Based Heart Disease Detection")
st.markdown("---")

patient_name = st.text_input("👤 Patient Name टाका:", placeholder="उदा. Rahul Sharma")
uploaded = st.file_uploader("📁 2D Echo Image निवडा...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(img_array)
        heatmap_img, overlay_img = get_heatmap(img_array)

    result = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown("### 🖼️ Image Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image.resize((224, 224)), caption="📷 Original Image", use_container_width=True)
    with col2:
        st.image(heatmap_img, caption="🔥 Heatmap", use_container_width=True)
    with col3:
        st.image(overlay_img, caption="🖼️ Overlay", use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🔍 Detected Disease: **{result}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
    with col2:
        st.markdown("### 📈 All Predictions:")
        for i, cls in enumerate(CLASSES):
            st.progress(float(prediction[0][i]), text=f"{cls}: {prediction[0][i]*100:.2f}%")

    st.markdown("---")

    if patient_name:
        pdf_buffer = generate_pdf(
            patient_name, result, confidence,
            prediction, image, heatmap_img, overlay_img
        )
        st.download_button(
            label="📄 PDF Report Download करा",
            data=pdf_buffer,
            file_name=f"Heart_Report_{patient_name}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("⚠️ PDF Report साठी Patient Name टाका!")