import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

@st.cache_resource(show_spinner=True)
def load_cnn_model():
    model = load_model('deepfake_cnn_model.h5')
    return model

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    image = image.convert('RGB')
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Deepfake CNN 模型圖片預測")

    model = load_cnn_model()

    uploaded_file = st.file_uploader("上傳圖片 (jpg/png)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='上傳圖片', use_container_width=True)

        img_input = preprocess_image(image)
        prediction = model.predict(img_input)[0][0]
        confidence = float(prediction)

        label = "Deepfake" if confidence >= 0.5 else "Real"

        st.markdown("### 預測結果")
        st.write(f"模型判斷：**{label}**")
        st.write(f"信心分數：**{confidence:.2%}**")

        confidence_clamped = max(0.0, min(confidence, 1.0))
        st.progress(int(confidence_clamped * 100))

if __name__ == "__main__":
    main()
