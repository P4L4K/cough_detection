import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tensorflow.keras.models import load_model
import joblib  # to load the saved preprocessor

# ----------------------------
# CONFIGURATION
# ----------------------------
TARGET_SR = 16000         # sample rate of model training data
TARGET_DURATION = 4.68    # seconds
MAX_SEGMENTS = 30         # for embedding padding/truncation
THRESHOLD = 0.5           # probability threshold

# ----------------------------
# LOAD MODELS AND PREPROCESSOR
# ----------------------------
# Load trained LSTM + metadata model
model = load_model("yamnet_88.keras")

# Load YAMNet (local cache first)
try:
    yamnet_model = tf.saved_model.load("yamnet_saved_model")
    print("Loaded YAMNet from local directory ✅")
except:
    print("Downloading YAMNet from TF Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    tf.saved_model.save(yamnet_model, "yamnet_saved_model")  # cache for next time

# Load saved metadata preprocessor
preprocessor = joblib.load("preprocessor_saved.pkl")

# ----------------------------
# AUDIO PREPROCESSING FUNCTION (in-memory)
# ----------------------------
def preprocess_audio_in_memory(file_path):
    # 1️⃣ Load audio
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # 2️⃣ Resample if needed
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # 3️⃣ Normalize amplitude
    y = y / (np.max(np.abs(y)) + 1e-6)

    # 4️⃣ Trim or pad to target duration
    target_length = int(TARGET_DURATION * sr)
    if len(y) > target_length:
        y = y[:target_length]   # trim
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))  # pad with zeros

    return y, sr

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_cough(wav_path, meta_dict):
    # Preprocess audio in-memory
    waveform, sr = preprocess_audio_in_memory(wav_path)
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Compute YAMNet embeddings
    _, embeddings, _ = yamnet_model(waveform)
    embeddings = embeddings.numpy()

    # Pad or truncate embeddings to MAX_SEGMENTS
    if embeddings.shape[0] < MAX_SEGMENTS:
        embeddings = np.pad(embeddings, ((0, MAX_SEGMENTS - embeddings.shape[0]), (0, 0)), mode='constant')
    else:
        embeddings = embeddings[:MAX_SEGMENTS]

    # Process metadata
    meta_input = preprocessor.transform(pd.DataFrame([meta_dict])).astype('float32')

    # Predict cough probability
    pred = model.predict([embeddings[np.newaxis, :, :], meta_input], verbose=0)
    prob = float(pred[0, 0])

    # Convert probability to label
    label = "Cough" if prob >= THRESHOLD else "Not Cough"
    return prob, label

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
meta_example = {'age': 68, 'gender': 'male', 'respiratory_condition': 'FALSE'}
wav_file = "/mnt/e/COUGH_DETECTION/2_test.wav"  # raw audio file

probability, label = predict_cough(wav_file, meta_example)
print(f"Predicted cough probability: {probability:.4f}")
print(f"Final Prediction: {label}")
