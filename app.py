from flask import Flask, request, jsonify, render_template
import joblib
import os
import cv2
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from werkzeug.utils import secure_filename

app = Flask(__name__)
    
# Load the trained text emotion detection model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model_text = joblib.load("models/model_textlr.pkl")
text_classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# Load the speech emotion detection model
speech_model_path = 'models/speech_svm_model.pkl'
speech_model = joblib.load(speech_model_path)
scaler = joblib.load('models/speech_scaler.pkl')  # Speech scaler
speech_classes = ['angry', 'sad', 'disgust', 'fear', 'happy', 'neutral']

# Initialize OneHotEncoder
encoder = OneHotEncoder()
encoder.categories_ = [np.array(speech_classes)]

# Load the image emotion detection model
model_image = joblib.load("models/image_lightgbm_model.pkl")
image_scaler = joblib.load("models/image_scaler.pkl")
image_pca = joblib.load("models/image_model_pca.pkl")
image_label_encoder = joblib.load("models/image_label_encoder.pkl")

# Define classes for image emotion detection
image_classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Mapping of emotions to emojis
emotion_to_emoji = {
    'angry': '😠',
    'anger': '😠',
    'fear': '😨',
    'happy': '😃',
    'neutral': '😐',
    'sad': '😔',
    'sadness': '😔',
    'surprise': '😱',
    'love': '😘',
    'disgust': '🤢',
    'joy': '😃',
}

# Feature extraction functions for audio
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def preprocess_speech(file_path):
    data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    features = scaler.transform(features.reshape(1, -1))
    return features

# Initialize HOG descriptor
hog = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)

def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Skip invalid images
    img = cv2.resize(img, (48, 48))
    features = hog.compute(img)
    return features.flatten()

def preprocess_image(file_path):
    features = extract_hog_features(file_path)
    if features is not None:
        features = image_scaler.transform([features])  # Apply StandardScaler
        features = image_pca.transform(features)  # Apply PCA
        return features
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')  # Show HTML page

    try:
        input_type = request.form.get("input_type", None)
        file = request.files.get("file", None)
        text = request.form.get("text", None)

        # Initialize variables to avoid unbound errors
        emotion = None
        emoji = ""
        confidence_scores = []
        labels = []

        if input_type == "text" and text:
            # Vectorize input text
            text_vectorized = vectorizer.transform([text]).toarray()
            prediction_proba = model_text.predict_proba(text_vectorized)[0]
            prediction = np.argmax(prediction_proba)
            emotion = label_encoder.inverse_transform([prediction])[0]
            emoji = emotion_to_emoji.get(emotion, '')
            confidence_scores = prediction_proba.tolist()
            labels = text_classes


        elif input_type == "image" and file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)

            features = preprocess_image(file_path)
            if features is not None:
                prediction_proba = model_image.predict_proba(features)[0]
                prediction = np.argmax(prediction_proba)
                emotion = image_label_encoder.inverse_transform([prediction])[0]
                emoji = emotion_to_emoji.get(emotion, '')
                confidence_scores = prediction_proba.tolist()
                labels = image_classes

        elif input_type == "speech" and file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)

            features = preprocess_speech(file_path)
            prediction_proba = speech_model.predict_proba(features)[0]
            prediction = np.argmax(prediction_proba)
            emotion = speech_classes[prediction]
            emoji = emotion_to_emoji.get(emotion, '')
            confidence_scores = prediction_proba.tolist()
            labels = speech_classes

        if emotion is None:
            return jsonify({"error": "Invalid input type or missing input"}), 400

        return render_template('index.html', prediction=emotion, emoji=emoji, confidence_scores=confidence_scores, labels=labels)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
