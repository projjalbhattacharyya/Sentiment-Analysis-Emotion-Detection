from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import joblib
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = Flask(__name__)

# Load the image emotion detection model
image_model_path = 'models/model_image.pkl'
image_model = joblib.load(image_model_path)

# Define classes for image emotion detection
image_classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the speech emotion detection model
speech_model_path = 'models/model_speech.pkl'
speech_model = joblib.load(speech_model_path)

# Define classes for speech emotion detection
speech_classes = ['angry', 'sad', 'disgust', 'fear', 'happy', 'neutral']

# Load the scaler used during training
scaler = joblib.load('models/scaler.pkl')  # Ensure scaler is saved and loaded correctly

# Initialize OneHotEncoder
encoder = OneHotEncoder()
encoder.categories_ = [np.array(speech_classes)]

# Load the text model
text_model_path = 'models/model_text.pkl'
text_model = joblib.load(text_model_path)
# Define classes for text emotion detection
text_classes = ['sad', 'happy', 'love', 'angry', 'fear', 'surprise']

# Load the tokenizer
with open('models/tokenizer.json', 'r') as file:
    tokenizer_json = json.load(file)
tokenizer = tokenizer_from_json(tokenizer_json)

# Define maximum length for padding
MAX_LEN = 50

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
    features = np.expand_dims(features, axis=2)
    return features

def preprocess_image(file_path):
    img = load_img(file_path, target_size=(48, 48), color_mode="grayscale")  # Adjust target_size as per your model
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
    return padded_sequences

# Mapping of emotions to emojis
emotion_to_emoji = {
    'angry': '😠',
    'fear': '😨',
    'happy': '😃',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲',
    'love': '❤️',
    'disgust': '🤢',
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form.get('input_type', None)
        file = request.files.get('file', None)
        text = request.form.get('text', None)

        if input_type == 'image' and file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)
            
            # Process image
            img = preprocess_image(file_path)
            prediction = image_model.predict(img)
            emotion = image_classes[np.argmax(prediction)]

        elif input_type == 'speech' and file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)

            # Process speech
            features = preprocess_speech(file_path)
            prediction = speech_model.predict(features)
            avg_prediction = np.mean(prediction, axis=0)
            emotion = speech_classes[np.argmax(avg_prediction)]

        elif input_type == 'text' and text:
            # Process text
            text_data = preprocess_text(text)
            prediction = text_model.predict(text_data)
            emotion = text_classes[np.argmax(prediction)]

        else:
            return jsonify({'error': 'Unsupported input type or missing input'})
        
        # Get the emoji for the predicted emotion
        emoji = emotion_to_emoji.get(emotion, '')
        return render_template('index.html', prediction=emotion, emoji=emoji)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)