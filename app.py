import cv2
import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from module import preprocess_image, CannyFeatureExtractor, GLCMFeatureExtractor, SVM_Scratch

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Deteksi Emosi Wajah", layout="centered")

def load_model():
    with open("svm_model.pkl", "rb") as f:
        return pickle.load(f)
    
def load_scaler():
    with open("svm_scaler.pkl", "rb") as f:
        return pickle.load(f)
    
def load_encoder():
    with open("svm_encoder.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
encoder = load_encoder()

print("Model loaded successfully.")
print(hasattr(model, 'alphas'))  # → True kalau sudah training
print(hasattr(model, 'project')) # → True kalau kelas lengkap

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
.main .block-container {
    max-width: 1200px;
    padding-left: 3rem;
    padding-right: 3rem;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #FFF;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #888;
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
.container-style {
    padding: 20px;
    margin-top: 20px;
    background-color: #111;  /* dark theme */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
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
        print(f"Canny features shape: {canny_features.shape}, range: {canny_features.min()}-{canny_features.max()}")
        
        # Step 4: Ekstraksi fitur GLCM
        glcm_extractor = GLCMFeatureExtractor()
        glcm_features_list = glcm_extractor.extract_features(preprocessed_img)
        
        # Step 5: Gabungkan fitur GLCM
        glcm_features = []
        for glcm_dict in glcm_features_list:
            glcm_features.extend(list(glcm_dict.values()))
        glcm_features = np.array(glcm_features)
        print(f"GLCM features shape: {glcm_features.shape}, range: {glcm_features.min()}-{glcm_features.max()}")
        
        # Step 6: Gabungkan semua fitur
        combined_features = np.concatenate([canny_features, glcm_features])
        print(f"Combined features shape: {combined_features.shape}")
        features = combined_features.reshape(1, -1)
        
        # Step 7: Normalisasi fitur (jika diperlukan)
        print(f"Before scaling - features range: {features.min()}-{features.max()}")
        features = scaler.transform(features)
        print(f"After scaling - features range: {features.min()}-{features.max()}")
        
        # Step 8: Prediksi dengan model
        label = model.predict(features)

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
        uploaded_content = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(uploaded_content), dtype=np.uint8)
        
        # Decode gambar
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Periksa apakah gambar berhasil dibaca
        if image is None:
            st.error("Gagal membaca gambar. Pastikan file gambar valid.")
        else:
            # Convert BGR to RGB untuk ditampilkan di Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with st.container():
                col1, col2 = st.columns([1, 1])

                # Kolom kiri untuk gambar kecil
                with col1:
                    st.image(image_rgb, caption='Gambar yang diunggah', width=300)

                # Kolom kanan untuk hasil prediksi
                with col2:
                    with st.spinner("Menganalisis emosi..."):
                        # Panggil model deteksi emosi
                        predicted_label, probabilities = predict_emotion(image)
                        label_name = encoder.inverse_transform([predicted_label])[0]
                        
                        # Tampilkan hasil prediksi
                        st.markdown("### Hasil Deteksi Emosi")
                        emoji = EMOTION_EMOJI.get(label_name, "")
                        st.success(f"Emosi terdeteksi: {emoji} {label_name}")
                        
                        # Persentase keyakinan
                        st.markdown("#### Persentase Keyakinan:")
                        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                        
                        for emotion, score in sorted_probs:
                            label_emotion = encoder.inverse_transform([emotion])[0]
                            emoji = EMOTION_EMOJI.get(label_emotion, "")
                            bar_color = "green" if label_emotion == label_name else "blue"
                            st.markdown(f"""
                                <div style='margin-bottom:8px'>
                                    <b>{emoji} {label_emotion}</b>: {score * 100:.2f}%
                                    <div style='background:#eee; border-radius:8px; overflow:hidden'>
                                        <div style='width:{score*100:.2f}%; background:{bar_color}; height:12px'></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error("Terjadi kesalahan saat memproses gambar.")
        st.error(f"Detail kesalahan: {str(e)}")
        st.info("Periksa apakah gambar berisi wajah yang terdeteksi dengan jelas.")