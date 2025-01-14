# Bird Species Classifier Web Application 🦜

A modern web application for real-time bird species classification from audio recordings using deep learning models.

## Features ✨

- Upload and process audio files (supports .wav, .mp3, and .flac formats)
- Multiple pre-trained models to choose from:
  - MobileNetV2-based models (Chroma and CQT variants)
  - VGG16-based models (Chroma and CQT variants)
- Real-time audio feature visualization:
  - Mel Spectrogram
  - Constant-Q Transform (CQT)
  - Mel-frequency Cepstral Coefficients (MFCC)
  - Chromagram
- Window-based prediction system for temporal analysis
- Responsive and intuitive user interface
- Efficient processing with background noise handling

## Supported Bird Species 🐦

The system can classify 21 different categories including:
- Acrocephalus arundinaceus
- Acrocephalus melanopogon
- Acrocephalus scirpaceus
- Alcedo atthis
- Anas platyrhynchos
- And many more...

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bird-species-classifier.git
cd bird-species-classifier
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install flask numpy pandas librosa matplotlib seaborn tensorflow scikit-learn flask-session
```

4. Create the necessary directories:
```bash
mkdir uploads
mkdir static/css
mkdir static/js
mkdir templates
mkdir models
```

5. Place your trained models in the `models` directory with the following naming convention:
- ChromaMNV21.keras
- ChromaMNV22.keras
- CQTMNV23.keras
- CQTMNV24.keras
- ChromaVGG1.keras
- ChromaVGG2.keras
- CQTVGG3.keras
- CQTVGG4.keras

## Usage 💻

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an audio file and select a model for classification

4. View the predictions and analyze the audio features for each time window

## Technical Details 🔧

### Backend
- Flask web framework
- TensorFlow for model inference
- Librosa for audio processing
- NumPy and Pandas for data handling
- Matplotlib for feature visualization

### Frontend
- Pure HTML, CSS, and JavaScript
- Responsive design with modern UI components
- Real-time feature visualization
- Interactive window selection

## File Structure 📁
```
bird-species-classifier/
├── app.py                 # Flask application
├── static/
│   ├── css/
│   │   └── test.css      # Styles
│   └── js/
│       └── trial.js      # Frontend logic
├── templates/
│   └── index.html        # Main page template
├── models/               # Trained models
└── uploads/             # Temporary file storage
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Thanks to all contributors and testers
- Bird sound datasets providers
- Deep learning model architecture developers
