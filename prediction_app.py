import streamlit as st
import tempfile
import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import joblib

# ----------------------------
# CONFIGURATION
# ----------------------------
TARGET_SR = 16000
TARGET_DURATION = 4.68
MAX_SEGMENTS = 30
THRESHOLD = 0.5

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    model = load_model("yamnet_88.keras")
    try:
        yamnet_model = tf.saved_model.load("yamnet_saved_model")
    except:
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        tf.saved_model.save(yamnet_model, "yamnet_saved_model")
    preprocessor = joblib.load("preprocessor_saved.pkl")
    return model, yamnet_model, preprocessor

model, yamnet_model, preprocessor = load_models()

# ----------------------------
# AUDIO PREPROCESSING FUNCTION
# ----------------------------
def preprocess_audio_in_memory(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = y / (np.max(np.abs(y)) + 1e-6)
    target_length = int(TARGET_DURATION * sr)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    return y, sr

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_cough(wav_path, meta_dict):
    waveform, sr = preprocess_audio_in_memory(wav_path)
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    embeddings = embeddings.numpy()
    if embeddings.shape[0] < MAX_SEGMENTS:
        embeddings = np.pad(embeddings, ((0, MAX_SEGMENTS - embeddings.shape[0]), (0, 0)), mode='constant')
    else:
        embeddings = embeddings[:MAX_SEGMENTS]
    meta_input = preprocessor.transform(pd.DataFrame([meta_dict])).astype('float32')
    pred = model.predict([embeddings[np.newaxis, :, :], meta_input], verbose=0)
    prob = float(pred[0, 0])
    label = "Cough" if prob >= THRESHOLD else "Not Cough"
    return prob, label

# ----------------------------
# STREAMLIT APP UI
# ----------------------------
st.title("Real-time Cough Detection")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

st.write("Enter metadata:")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["male", "female", "other"])
resp_condition = st.selectbox("Respiratory condition?", ["TRUE", "FALSE"])

if uploaded_file is not None:
    # Play uploaded audio
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Predict Cough"):
        # Convert any audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext != ".wav":
                y, sr = librosa.load(uploaded_file, sr=None, mono=True)
                sf.write(tmp_wav.name, y, sr, subtype='PCM_16')
            else:
                tmp_wav.write(uploaded_file.read())
            tmp_wav_path = tmp_wav.name

        # Prepare metadata
        meta_example = {"age": age, "gender": gender, "respiratory_condition": resp_condition}

        # Predict
        probability, label = predict_cough(tmp_wav_path, meta_example)
        st.write(f"**Predicted cough probability:** {probability:.4f}")
        st.write(f"**Final Prediction:** {label}")

        # Clean up temp file
        os.remove(tmp_wav_path)
