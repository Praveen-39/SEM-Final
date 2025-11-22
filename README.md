# Speech Emotion & Sarcasm Detection

A real-time speech emotion recognition system with advanced sarcasm detection capabilities, built using CNN-LSTM architecture, TensorFlow, and Streamlit.

## 🎯 Features

- **Emotion Detection**: Recognizes 8 different emotions (Happy, Sad, Angry, Fearful, Neutral, Surprised, Calm, Disgust)
- **Sarcasm Detection**: Advanced acoustic analysis to detect sarcasm in speech
- **Real-time Analysis**: Upload audio files and get instant results
- **Detailed Reports**: Download comprehensive analysis reports

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow 2.13.0** - Deep learning framework
- **Streamlit 1.28.0** - Web application framework
- **Librosa 0.10.1** - Audio analysis
- **CNN-LSTM Architecture** - Neural network model

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (if models are not included):
```bash
python model_training.py
```

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
Final Project/
├── app.py                 # Main Streamlit application
├── model_training.py      # Model training script
├── feature_extraction.py  # Audio feature extraction
├── sarcasm_detector.py    # Sarcasm detection module
├── requirements.txt       # Python dependencies
├── models/                # Trained model files
│   ├── emotion_model.h5
│   └── label_encoder.npy
└── README.md
```

## 🎓 Project Details

- **Project Type**: Final Year BSc CS AI&DS Project
- **Year**: 2025
- **Focus**: Speech Emotion Recognition with Sarcasm Detection

## 📝 License

This project is for educational purposes.

## 👤 Author

Your Name - Final Year BSc CS AI&DS Student

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the web framework
- Librosa for audio processing capabilities
