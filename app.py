"""
Speech Emotion Detection with Sarcasm Detection
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import tempfile
import os
from feature_extraction import extract_features
from sarcasm_detector import SarcasmDetector

# Page configuration
st.set_page_config(
    page_title="Speech Emotion & Sarcasm Detector",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .emotion-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
    }
    .sarcasm-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load emotion detection model and label encoder"""
    try:
        emotion_model = keras.models.load_model('models/emotion_model.h5')
        label_encoder_classes = np.load('models/label_encoder.npy', allow_pickle=True)
        return emotion_model, label_encoder_classes
    except Exception as e:
        st.error(f"⚠️ Models not found! Error: {e}")
        st.info("Please train the model first by running: `python model_training.py`")
        return None, None

# Initialize
emotion_model, label_classes = load_models()
sarcasm_detector = SarcasmDetector(threshold=0.6)

# Header
st.markdown('<div class="main-header">🎤 Speech Emotion & Sarcasm Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time emotion recognition with advanced sarcasm detection</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📊 Detection Parameters")
    confidence_threshold = st.slider(
        "Emotion Confidence Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Minimum confidence for emotion detection"
    )
    
    sarcasm_threshold = st.slider(
        "Sarcasm Detection Threshold",
        0.0, 1.0, 0.6, 0.05,
        help="Minimum probability for sarcasm detection"
    )
    
    st.markdown("---")
    
    st.subheader("📖 About")
    st.markdown("""
    This system uses:
    - **CNN-LSTM Model** for emotion recognition
    - **Acoustic Analysis** for sarcasm detection
    
    **Emotions Detected:**
    - 😊 Happy
    - 😢 Sad
    - 😠 Angry
    - 😨 Fearful
    - 😐 Neutral
    - 😲 Surprised
    - 😌 Calm
    - 🤢 Disgust
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** Your Name")
    st.markdown("**Project:** Final Year BSc CS AI&DS")

# Main content
st.markdown("### 🎵 Upload Audio File")

uploaded_file = st.file_uploader(
    "Choose an audio file (WAV, MP3, OGG, FLAC)",
    type=['wav', 'mp3', 'ogg', 'flac'],
    help="Upload a voice recording to analyze emotion and detect sarcasm"
)

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file)
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_file = tmp_file.name
    
    # Load audio
    try:
        audio_data, sample_rate = librosa.load(audio_file, sr=22050)
        
        st.success("✅ Audio loaded successfully!")
        
        # Audio info
        duration = len(audio_data) / sample_rate
        st.info(f"📊 **Duration:** {duration:.2f} seconds | **Sample Rate:** {sample_rate} Hz")
        
        # Analysis button
        if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
            
            if emotion_model is None:
                st.error("❌ Model not loaded. Please train the model first!")
            else:
                with st.spinner("🔄 Analyzing audio..."):
                    
                    # Extract features
                    features = extract_features(audio_file)
                    
                    if features is not None:
                        # Prepare for prediction
                        features = features.reshape(1, -1, 1)
                        
                        # Predict emotion
                        emotion_probs = emotion_model.predict(features, verbose=0)[0]
                        emotion_idx = np.argmax(emotion_probs)
                        predicted_emotion = label_classes[emotion_idx]
                        emotion_confidence = emotion_probs[emotion_idx]
                        
                        # Detect sarcasm
                        sarcasm_result = sarcasm_detector.detect_sarcasm(
                            audio_data,
                            sample_rate,
                            predicted_emotion
                        )
                        
                        # Update sarcasm threshold
                        sarcasm_detector.threshold = sarcasm_threshold
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        # Emotion results
                        with col1:
                            st.markdown("### 😊 Emotion Detection")
                            
                            emotion_emoji = {
                                'happy': '😊', 'sad': '😢', 'angry': '😠',
                                'fearful': '😨', 'neutral': '😐', 'surprised': '😲',
                                'calm': '😌', 'disgust': '🤢'
                            }
                            
                            emoji = emotion_emoji.get(predicted_emotion, '😐')
                            
                            st.markdown(f'<div class="result-box emotion-box">', unsafe_allow_html=True)
                            st.markdown(f"## {emoji} **{predicted_emotion.upper()}**")
                            st.progress(float(emotion_confidence))
                            st.metric("Confidence", f"{emotion_confidence * 100:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Top 3 emotions
                            with st.expander("📈 Top 3 Emotions"):
                                top_3_idx = np.argsort(emotion_probs)[-3:][::-1]
                                for idx in top_3_idx:
                                    emotion = label_classes[idx]
                                    prob = emotion_probs[idx]
                                    emoji = emotion_emoji.get(emotion, '😐')
                                    st.write(f"{emoji} **{emotion.capitalize()}**: {prob * 100:.1f}%")
                        
                        # Sarcasm results
                        with col2:
                            st.markdown("### 🎭 Sarcasm Detection")
                            
                            st.markdown(f'<div class="result-box sarcasm-box">', unsafe_allow_html=True)
                            
                            if sarcasm_result['is_sarcastic']:
                                st.markdown("## 🎭 **SARCASTIC**")
                                st.error("⚠️ Sarcasm detected!")
                            else:
                                st.markdown("## ✅ **NOT SARCASTIC**")
                                st.success("✓ No sarcasm detected")
                            
                            st.progress(float(sarcasm_result['probability']))
                            st.metric("Confidence", f"{sarcasm_result['confidence']:.1f}%")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Sarcasm indicators
                            if sarcasm_result['indicators']:
                                with st.expander("🔍 Sarcasm Indicators"):
                                    for indicator in sarcasm_result['indicators']:
                                        st.write(f"• {indicator}")
                        
                        # Detailed acoustic features
                        with st.expander("📈 Detailed Acoustic Features"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.metric("Pitch Std Dev", f"{sarcasm_result['acoustic_features']['pitch_std']:.2f}")
                                st.metric("Pitch Range", f"{sarcasm_result['acoustic_features']['pitch_range']:.2f}")
                            
                            with col_b:
                                st.metric("Energy Variance", f"{sarcasm_result['acoustic_features']['energy_variance']:.4f}")
                                st.metric("Tempo (BPM)", f"{sarcasm_result['acoustic_features']['tempo']:.1f}")
                        
                        # Summary
                        st.markdown("---")
                        st.markdown("### 📋 Summary")
                        
                        summary = f"""
                        **Detected Emotion:** {predicted_emotion.upper()} ({emotion_confidence * 100:.1f}% confidence)
                        
                        **Sarcasm Status:** {'🎭 DETECTED' if sarcasm_result['is_sarcastic'] else '✅ NOT DETECTED'} 
                        ({sarcasm_result['confidence']:.1f}% confidence)
                        
                        **Interpretation:** The speaker sounds **{predicted_emotion}** 
                        {'and appears to be speaking **sarcastically**.' if sarcasm_result['is_sarcastic'] else 'with **genuine emotion**.'}
                        """
                        
                        st.info(summary)
                        
                        # Download results
                        results_text = f"""
Speech Emotion & Sarcasm Detection Report
==========================================

Audio File: {uploaded_file.name}
Duration: {duration:.2f} seconds

EMOTION DETECTION
-----------------
Predicted Emotion: {predicted_emotion.upper()}
Confidence: {emotion_confidence * 100:.1f}%

Top 3 Emotions:
{chr(10).join([f"{i+1}. {label_classes[idx].capitalize()}: {emotion_probs[idx] * 100:.1f}%" for i, idx in enumerate(np.argsort(emotion_probs)[-3:][::-1])])}

SARCASM DETECTION
-----------------
Status: {'SARCASTIC' if sarcasm_result['is_sarcastic'] else 'NOT SARCASTIC'}
Confidence: {sarcasm_result['confidence']:.1f}%

Indicators:
{chr(10).join([f"- {ind}" for ind in sarcasm_result['indicators']]) if sarcasm_result['indicators'] else 'None'}

ACOUSTIC FEATURES
-----------------
Pitch Std Dev: {sarcasm_result['acoustic_features']['pitch_std']:.2f}
Pitch Range: {sarcasm_result['acoustic_features']['pitch_range']:.2f}
Energy Variance: {sarcasm_result['acoustic_features']['energy_variance']:.4f}
Tempo: {sarcasm_result['acoustic_features']['tempo']:.1f} BPM

INTERPRETATION
--------------
{summary}
                        """
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=results_text,
                            file_name=f"emotion_sarcasm_report_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
                    
                    else:
                        st.error("❌ Error extracting features from audio")
        
    except Exception as e:
        st.error(f"❌ Error loading audio: {e}")

else:
    # Instructions when no file uploaded
    st.info("""
    👋 **Welcome!** Please upload an audio file to get started.
    
    **How to use:**
    1. Click "Browse files" above
    2. Select a WAV, MP3, OGG, or FLAC file
    3. Click "Analyze Audio"
    4. View emotion and sarcasm detection results
    
    **Tips for best results:**
    - Use clear voice recordings
    - 3-5 seconds of speech works best
    - Minimize background noise
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using <b>CNN-LSTM Architecture</b> | <b>TensorFlow</b> | <b>Streamlit</b></p>
    <p>Speech Emotion Detection with Sarcasm Detection | Final Year Project 2025</p>
</div>
""", unsafe_allow_html=True)
