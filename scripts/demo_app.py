
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import os

# ==============================
# APP CONFIGURATION
# ==============================
st.set_page_config(
    page_title="🚀 Space Station Safety Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# STYLING
# ==============================
st.markdown(
    """
    <style>
        .main {
            background-color: #0b132b;
            color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #1c2541;
            color: #ffffff;
            border-radius: 6px;
            height: 2.8em;
            width: 100%;
            border: 1px solid #3a506b;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #3a506b;
            border: 1px solid #5bc0be;
        }
        .css-1v0mbdj img {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# APP TITLE SECTION
# ==============================
st.title("🛰️ Space Station Safety Object Detection")
st.markdown(
    """
    <div style="font-size:20px; color:#d9d9d9; margin-bottom:20px;">
        Upload a space station image to automatically detect critical safety equipment —
        including <b>oxygen tanks</b>, <b>fire extinguishers</b>, <b>alarms</b>, and more.
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model_path = r"C:\\Users\\Nitin\\Desktop\\Spacestation_Object_Detection\\runs\\detect\\train50\\weights\\best.pt"
    model = YOLO(model_path)
    return model

model = load_model()
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold:", 0.1, 0.9, 0.25)
iou = st.sidebar.slider("IoU Threshold:", 0.1, 0.9, 0.45)
device = st.sidebar.radio("Device:", ["cpu", "cuda"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("Trained on YOLOv8n | Fine-tuned for 50 epochs | mAP@0.5: 0.69")

# ==============================
# FILE UPLOAD
# ==============================
st.markdown("## 📁 Upload Space Station Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ==============================
# DETECTION
# ==============================
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("**Running Detection... Please wait ⚙️**")
        results = model(image, conf=confidence, iou=iou, device=device)
        result_image = results[0].plot()

        st.image(result_image, caption="✅ Detected Objects", use_column_width=True)

        st.markdown("### 📋 Detected Objects Summary")
        for box in results[0].boxes.data.tolist():
            cls_id = int(box[5])
            conf_val = round(box[4], 2)
            label = model.names[cls_id]
            st.markdown(f"• **{label}** — Confidence: `{conf_val}`")

    st.success("🎉 Detection Completed Successfully!")

else:
    st.info("👆 Please upload a space station image to begin detection.")

# ==============================
# FOOTER
# ==============================
st.markdown(
    """
    <hr>
    <center style='font-size:15px; color:#d3d3d3;'>
    Built with ❤️ using YOLOv8 and Streamlit<br>
    Developed by <b>Nitin</b> · B.Tech CSE (3rd Year)
    </center>
    """,
    unsafe_allow_html=True,
)