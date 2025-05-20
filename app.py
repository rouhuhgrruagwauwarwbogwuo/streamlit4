import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Deepfake 偵測", layout="centered")

# ====== 模型下載與載入 ======
@st.cache_resource
def load_custom_cnn_model():
    model_url = "https://huggingface.co/wuwuwu123123/newmodel/resolve/main/deepfake_cnn_model.h5"
    model_path = "deepfake_cnn_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("⬇️ 正在從 Hugging Face 下載自訂模型..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(model_url, headers=headers)
            if response.status_code != 200:
                st.error("❌ 模型下載失敗，請確認 Hugging Face 連結是否正確。")
                st.stop()
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("✅ 模型下載完成！")

    model = load_model(model_path)
    return model

# ====== 圖片預處理 ======
def preprocess_image(img: Image.Image, target_size=(128, 128)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = img_to_array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ====== 主程式 ======
def main():
    st.title("🧠 Deepfake 圖像偵測系統")
    st.markdown("上傳一張人臉圖片，我們將使用自訂 CNN 模型進行 Deepfake 分析。")

    uploaded_file = st.file_uploader("📷 上傳圖片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳圖片", use_container_width=True)

        # 載入模型
        model = load_custom_cnn_model()

        # 圖像預處理
        preprocessed_img = preprocess_image(image)

        # 預測
        prediction = model.predict(preprocessed_img)[0][0]
        label = "🟢 真實 Real" if prediction < 0.5 else "🔴 假的 Deepfake"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown("---")
        st.subheader("🔍 偵測結果")
        st.markdown(f"**判斷：{label}**")
        st.progress(float(confidence), text=f"信心分數：{confidence:.2%}")

if __name__ == "__main__":
    main()
