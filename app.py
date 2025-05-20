import streamlit as st
import numpy as np
import os
import cv2
import tempfile
from PIL import Image
from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖像偵測器")

# 載入模型
@st.cache_resource
def load_models():
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

    resnet_classifier = Sequential([resnet_model, Dense(1, activation='sigmoid')])
    efficientnet_classifier = Sequential([efficientnet_model, Dense(1, activation='sigmoid')])
    xception_classifier = Sequential([xception_model, Dense(1, activation='sigmoid')])

    return {
        'ResNet50': resnet_classifier,
        'EfficientNet': efficientnet_classifier,
        'Xception': xception_classifier
    }

# 使用 OpenCV 提取人臉
def extract_face_opencv(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = np.array(pil_img)[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# 高通濾波
def high_pass_filter(img):
    img_np = np.array(img)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered_img = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(filtered_img)

# CLAHE + 銳化
def apply_clahe_sharpen(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(img_clahe, (0, 0), 3)
    sharpened = cv2.addWeighted(img_clahe, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

# 預處理圖像
def preprocess_image(img, model_name):
    img = apply_clahe_sharpen(img)
    img = high_pass_filter(img)

    if model_name == 'Xception':
        img = img.resize((299, 299))
        img_array = np.array(img).astype(np.float32)
        return preprocess_xception(img_array)
    else:
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        if model_name == 'ResNet50':
            return preprocess_resnet(img_array)
        elif model_name == 'EfficientNet':
            return preprocess_efficientnet(img_array)
    return img_array

# 單模型預測
def predict_model(models, img):
    predictions = []
    for name, model in models.items():
        input_data = preprocess_image(img, name)
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data, verbose=0)
        predictions.append(prediction[0][0])
    return predictions

# 集成預測
def stacking_predict(models, img, threshold=0.55):  # 設定閥值為0.55
    preds = predict_model(models, img)
    avg = np.mean(preds)
    label = "Deepfake" if avg > threshold else "Real"
    return label, avg

# 顯示預測結果
def show_prediction(img, models, threshold=0.55):  # 設定閥值為0.55
    label, confidence = stacking_predict(models, img, threshold)
    st.image(img, caption="輸入圖像", use_container_width=True)
    st.subheader(f"預測結果：**{label}**")
    st.markdown(f"信心分數：**{confidence:.2f}**")

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], confidence, color='green' if label == "Real" else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('信心分數')
    st.pyplot(fig)

# 主體
models = load_models()
tab1, tab2 = st.tabs(["🖼️ 圖像偵測", "🎥 影片偵測"])

with tab1:
    st.header("上傳圖像進行 Deepfake 偵測")
    uploaded_image = st.file_uploader("選擇一張圖像", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖像", use_container_width=True)

        face_img = extract_face_opencv(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到人臉", width=300)
            show_prediction(face_img, models, threshold=0.55)
        else:
            st.info("⚠️ 沒偵測到人臉，使用整張圖像預測")
            show_prediction(pil_img, models, threshold=0.55)

with tab2:
    st.header("影片偵測（處理前幾幀）")
    uploaded_video = st.file_uploader("選擇一段影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 正在分析影片...（取前 10 幀）")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        shown = False
        max_frames = 10
        frame_confidences = []

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 3 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                face_img = extract_face_opencv(pil_frame)
                if face_img:
                    label, confidence = stacking_predict(models, face_img)
                    frame_confidences.append((label, confidence))
                    if not shown:
                        st.image(pil_frame, caption=f"幀 {frame_idx+1}")
                        st.write(f"信心分數：{confidence:.2f}")
                        st.write(f"預測結果：{label}")
                        shown = True
                else:
                    frame_confidences.append(("No face", 0.0))
            frame_idx += 1

        cap.release()
        st.write(f"分析結果：{frame_confidences}")
