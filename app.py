from flask import Flask, request, jsonify, render_template, session
import os
from io import BytesIO
import base64
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

import pickle
from uuid import uuid4

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TEMP = ''

from flask_session import Session

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

birds = ["Acrocephalus arundinaceus", "Acrocephalus melanopogon", "Acrocephalus scirpaceus",
         'Alcedo atthis', 'Anas platyrhynchos', 'Anas strepera', 'Ardea purpurea',
         'Botaurus stellaris', 'Charadrius alexandrinus', 'Ciconia ciconia',
         'Circus aeruginosus', 'Coracias garrulus', 'Dendrocopos minor',
         'Fulica atra', 'Gallinula chloropus', 'Himantopus himantopus',
         'Ixobrychus minutus', 'Motacilla flava', "No bird",
         'Porphyrio porphyrio', "Tachybaptus ruficollis"]

model_paths = {
    "ChromaMNV21": "models/ChromaMNV21.keras",
    "ChromaMNV22": "models/ChromaMNV22.keras",
    "CQTMNV23": "models/CQTMNV23.keras",
    "CQTMNV24": "models/CQTMNV24.keras",
    "ChromaVGG1": "models/ChromaVGG1.keras",
    "ChromaVGG2": "models/ChromaVGG2.keras",
    "CQTVGG3": "models/CQTVGG3.keras",
    "CQTVGG4": "models/CQTVGG4.keras"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_species():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if 'model' not in request.form:
        return jsonify({"error": "No model specified"}), 400

    model = request.form['model']

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            npyfile = convertnpy(app.config['UPLOAD_FOLDER'], file.filename)
            features = Extraction(npyfile, window_size=3).featuredictionary
            predictions = prediction(features, model)
            print("IM FINISHED")

            session_id = str(uuid4())
            file_path_pkl = os.path.join('uploads', f"{session_id}_features.pkl")
            with open(file_path_pkl, 'wb') as f:
                pickle.dump(features, f)

            session['current_features_file'] = file_path_pkl

            try:
                os.remove(file_path)
                os.remove(npyfile)
            except OSError:
                pass
            
            return jsonify(predictions)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up files
            try:
                os.remove(file_path)
                os.remove(npyfile)
            except OSError:
                pass

@app.route('/features', methods=['POST'])
def get_features():
    data = request.get_json()
    window = data.get('window', 0)
    
    if 'current_features_file' not in session:
        return jsonify({"error": "No features available"}), 400

    file_path = session.get('current_features_file')

    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    
    try:
        # Generate plots and convert to base64
        feature_images = {}
        for feature_name in ['melspectrogram', 'cqt', 'mfcc', 'chroma']:
            plt.figure(figsize=(10, 5))
            feature_data = features[feature_name][window]
            
            if feature_name == 'melspectrogram':
                librosa.display.specshow(feature_data, sr=22050, x_axis='time', y_axis='mel')
            elif feature_name == 'cqt':
                librosa.display.specshow(feature_data, sr=22050, x_axis='time', y_axis='cqt_note')
            elif feature_name == 'chroma':
                librosa.display.specshow(feature_data, sr=22050, x_axis='time', y_axis='chroma')
            else:  # mfcc
                librosa.display.specshow(feature_data, sr=22050, x_axis='time', y_axis='mel')
            
            plt.colorbar(format='%+2.0f dB')
            plt.title(feature_name.upper(), fontsize=16, fontweight='bold')
            
            # Convert plot to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            feature_images[feature_name] = image_base64
            
            plt.close()

        return jsonify(feature_images)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def convertnpy(path, file):
    sr = 22050
    filepath = os.path.join(path, file)
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    new_filename = file.replace('.mp3', '.npy')
    filename = f'{path}/{new_filename}'
    np.save(filename, audio)
    return filename

class Extraction:
    def __init__(self, npyfile, window_size=3, overlap=0.5, sr=22050, n_mels=128, n_mfcc=20, 
                 n_chroma=12, n_cqt=84, hoplength=256, 
                 features=['melspectrogram','cqt','mfcc', 'chroma'], normalize=True):
        self.npyfile = npyfile
        self.window_size = window_size
        self.overlap = overlap
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_cqt = n_cqt
        self.hoplength = hoplength
        self.normalize = normalize
        self.features = features
        self.featuredictionary = self.feature_extraction(self.npyfile, window_size=self.window_size)
    
    def normalize_audio(self, audio):
        return (audio - np.min(audio)) / (np.max(audio) - np.min(audio))
    
    def generate_pink_noise(self, num_samples):
        white_noise = np.random.randn(num_samples)
        X = np.fft.rfft(white_noise)
        S = np.arange(1, len(X) + 1)
        pink_noise = np.fft.irfft(X / S)
        if len(pink_noise) < num_samples:
            pink_noise = np.pad(pink_noise, (0, num_samples - len(pink_noise)), mode='constant')
        elif len(pink_noise) > num_samples:
            pink_noise = pink_noise[:num_samples]
        return self.normalize_audio(pink_noise)
    
    def pad_with_noise(self, audio_data, window_length, window_samples):
        current_length = librosa.get_duration(y=audio_data, sr=self.sr)
        if current_length > window_length:
            return audio_data
        target_length_samples = int(window_length * self.sr)
        padding_length_samples = target_length_samples - window_samples
        pink_noise = self.generate_pink_noise(padding_length_samples)
        return np.concatenate([audio_data, pink_noise])

    def extract_mfcc(self, window):
        mfcc = librosa.feature.mfcc(y=window, sr=self.sr, n_mfcc=self.n_mfcc, hop_length=self.hoplength)
        return librosa.util.normalize(mfcc) if self.normalize else mfcc

    def extract_chroma(self, window):
        chroma = librosa.feature.chroma_stft(y=window, sr=self.sr, n_chroma=self.n_chroma, 
                                           hop_length=self.hoplength)
        return librosa.util.normalize(chroma) if self.normalize else chroma

    def extract_cqt(self, window):
        cqt = librosa.cqt(y=window, sr=self.sr, hop_length=self.hoplength, n_bins=self.n_cqt)
        return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    def extract_melspectrogram(self, window):
        mel = librosa.feature.melspectrogram(y=window, sr=self.sr, n_mels=self.n_mels, 
                                           hop_length=self.hoplength)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return librosa.util.normalize(mel_db) if self.normalize else mel_db

    def feature_extraction(self, npyfile, window_size):
        features_dict = {item: [] for item in self.features}
        audio = np.load(npyfile)
        
        if len(audio) < 512:
            return None
        
        sample = self.normalize_audio(audio)
        sample = self.pad_with_noise(sample, window_length=self.window_size, window_samples=len(sample))
        
        window_samples = int(window_size * self.sr)
        hop_samples = int(window_samples * (1 - self.overlap))
        
        audio_windows = librosa.util.frame(sample, frame_length=window_samples, 
                                         hop_length=hop_samples).T
        
        for window in audio_windows:
            if not np.isfinite(window).all():
                continue
            
            if len(window) < window_samples:
                if len(window) < 512*2:
                    continue
                else:
                    window = self.pad_with_noise(window, window_length=window_size, 
                                               window_samples=len(window))
            
            for feature in self.features:
                extract = f"extract_{feature}"
                if hasattr(self, extract) and callable(func := getattr(self, extract)):
                    features_dict[feature].append(func(window))
        
        for key in features_dict:
            features_dict[key] = np.array(features_dict[key])
        
        return features_dict

def tile_and_crop(feature, target_size):
    tiled = np.tile(feature, (1, target_size // feature.shape[1] + 1, 1))
    return tiled[:, :target_size, :]

def prediction(features_dict, model_name):
    value = {}
    key = 'cqt' if "CQT" in model_name else 'chroma'
    
    model = load_model(model_paths[model_name])
    target_size = 128
    
    mfcc_tiled = tile_and_crop(features_dict['mfcc'], target_size)
    final = tile_and_crop(features_dict[key], target_size)
    features = np.stack((features_dict['melspectrogram'], mfcc_tiled, final), axis=-1)
    
    predictions = model.predict(features)
    
    if len(predictions) > 1:
        for i, window in enumerate(predictions):
            predicted_class = np.argmax(window)
            species = birds[predicted_class]
            value[f'Window {i} @ {i*1.5}s-{(i+1)*1.5}s'] = species
    else:
        predicted_class = np.argmax(predictions)
        species = birds[predicted_class]
        value['Window 0'] = species
    
    return value

if __name__ == '__main__':
    app.run(debug=True)