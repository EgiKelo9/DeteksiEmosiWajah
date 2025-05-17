import cv2
import pickle
import numpy as np
import streamlit as st
from ipynb.fs.full.svm_scratch_cv import get_model
from sklearn.preprocessing import StandardScaler
from module import preprocess_image, CannyFeatureExtractor, GLCMFeatureExtractor, SVM_Scratch

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Deteksi Emosi Wajah", layout="centered")

# @st.cache_resource
# def load_model():
#     with open("svm_model.pkl", "rb") as f:
#         model = SVM_Scratch()
#         model_data = pickle.load(f)
#         model.__dict__.update(model_data)
#     return model

# model = load_model()
model = get_model()

print(type(model))
print(hasattr(model, 'predict'))

# Emoji untuk setiap label emosi
EMOTION_EMOJI = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😱",
    "surprise": "😲",
    "neutral": "😐"
}

# CSS untuk styling dan animasi loader
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #3E64FF;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}
.loader {
    border: 4px solid #f3f3f3;
    border-radius: 50%;
    border-top: 4px solid #3E64FF;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# Judul dan subjudul
st.markdown('<div class="title">Deteksi Emosi Wajah Dengan SVM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unggah gambar wajah untuk mengklasifikasi emosi</div>', unsafe_allow_html=True)

# Fungsi untuk memprediksi emosi dari gambar
def predict_emotion(image):
    try:
        # Step 1: Pastikan gambar valid
        if image is None or image.size == 0:
            raise ValueError("Gambar tidak valid")
        
        # Step 2: Preprocessing gambar
        # Gunakan fungsi dari module.py
        preprocessed_img = preprocess_image(image)
        
        # Step 3: Ekstraksi fitur Canny
        canny_extractor = CannyFeatureExtractor()
        canny_features = canny_extractor.extract_features(preprocessed_img)
        
        # Step 4: Ekstraksi fitur GLCM
        glcm_extractor = GLCMFeatureExtractor()
        glcm_features_list = glcm_extractor.extract_features(preprocessed_img)
        
        # Step 5: Gabungkan fitur GLCM
        glcm_features = []
        for glcm_dict in glcm_features_list:
            glcm_features.extend(list(glcm_dict.values()))
        glcm_features = np.array(glcm_features)
        
        # Step 6: Gabungkan semua fitur
        combined_features = np.concatenate([canny_features, glcm_features])
        features = combined_features.reshape(1, -1)
        
        # Step 7: Normalisasi fitur (jika diperlukan)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Step 8: Prediksi dengan model
        label = model.predict(features)[0]

        # Step 9: Menghitung probabilitas jika tersedia
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features)[0]
                class_probs = dict(zip(model.classes, probs))
            elif hasattr(model, 'decision_function'):
                decisions = model.decision_function(features)
                if decisions.ndim == 1:
                    pos_class_probs = 1 / (1 + np.exp(-decisions))
                    class_probs = {model.classes[0]: 1 - pos_class_probs[0],
                                model.classes[1]: pos_class_probs[0]}
                else:
                    scores = decisions[0]
                    scores = scores - np.max(scores)
                    exp_scores = np.exp(scores)
                    probs = exp_scores / np.sum(exp_scores)
                    class_probs = dict(zip(model.classes, probs))
        except Exception as e:
            st.warning(f"Tidak dapat menghitung probabilitas: {str(e)}")
            class_probs = {str(label): 1.0}  # fallback label string

        return label, class_probs

        
    except Exception as e:
        st.error(f"Error dalam predict_emotion: {str(e)}")
        # Tampilkan traceback untuk debugging
        import traceback
        st.code(traceback.format_exc())
        raise e

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar wajah", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Proses gambar jika diunggah
if uploaded_file is not None:
    try:
        # Baca file ke dalam bytes array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Decode gambar
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Periksa apakah gambar berhasil dibaca
        if image is None:
            st.error("Gagal membaca gambar. Pastikan file gambar valid.")
        else:
            # Convert BGR to RGB untuk ditampilkan di Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption='Gambar yang diunggah', use_container_width=True)
            
            # Tampilkan animasi loading
            with st.spinner("Menganalisis emosi..."):
                # Panggil model deteksi emosi
                predicted_label, probabilities = predict_emotion(image)
                
                # Tampilkan hasil prediksi
                st.markdown("### Hasil Deteksi Emosi")
                emoji = EMOTION_EMOJI.get(predicted_label, "")
                st.success(f"Emosi terdeteksi: {emoji} {predicted_label}")
                
                # Persentase keyakinan
                st.markdown("#### Persentase Keyakinan:")
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                for emotion, score in sorted_probs:
                    emoji = EMOTION_EMOJI.get(emotion, "")
                    bar_color = "green" if emotion == predicted_label else "blue"
                    st.markdown(f"""
                        <div style='margin-bottom:8px'>
                            <b>{emoji} {emotion}</b>: {round(score * 100, 2)}%
                            <div style='background:#eee; border-radius:8px; overflow:hidden'>
                                <div style='width:{score*100:.2f}%; background:{bar_color}; height:12px'></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error("Terjadi kesalahan saat memproses gambar.")
        st.error(f"Detail kesalahan: {str(e)}")
        st.info("Periksa apakah gambar berisi wajah yang terdeteksi dengan jelas.")