"""
Audio Feature Extraction Module
Extracts MFCC, Chroma, Mel, Contrast, Tonnetz features from audio
"""

import librosa
import numpy as np
import soundfile as sf

def extract_features(audio_path, sr=22050, duration=3):
    """
    Extract comprehensive audio features for emotion and sarcasm detection
    
    Parameters:
    -----------
    audio_path : str
        Path to audio file (.wav, .mp3, etc.)
    sr : int
        Sample rate (default: 22050 Hz)
    duration : int
        Duration to process in seconds (default: 3)
    
    Returns:
    --------
    features : numpy.ndarray
        Array of extracted features (193 features)
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Feature 1: MFCC (Mel-frequency cepstral coefficients) - 40 features
        # Captures the power spectrum of sound - crucial for emotion
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        # Feature 2: Chroma - 12 features
        # Captures pitch and harmony - useful for detecting tone changes
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        
        # Feature 3: Mel Spectrogram - 128 features
        # Time-frequency representation of audio
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        # Feature 4: Spectral Contrast - 7 features
        # Measures difference between peaks and valleys in spectrum
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
        
        # Feature 5: Tonnetz (Tonal Centroid Features) - 6 features
        # Represents harmonic content
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)
        
        # Feature 6: Zero Crossing Rate - 1 feature
        # How often the signal changes from positive to negative
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Feature 7: Spectral Rolloff - 1 feature
        # Frequency below which 85% of energy is contained
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
        
        # Combine all features: 40 + 12 + 128 + 7 + 6 + 1 + 1 = 195 features
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, zcr, rolloff])
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def extract_emotion_features(audio, sr=22050):
    """
    Simplified feature extraction for real-time emotion detection
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Raw audio data
    sr : int
        Sample rate
    
    Returns:
    --------
    features : numpy.ndarray
        Combined feature array
    """
    # Extract only the most important features for speed
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    
    return np.hstack([mfcc, chroma, mel])


# Test function
if __name__ == "__main__":
    # Test with a sample audio file
    test_file = "test_audio.wav"
    
    try:
        features = extract_features(test_file)
        if features is not None:
            print(f"✅ Feature extraction successful!")
            print(f"Feature shape: {features.shape}")
            print(f"Total features: {len(features)}")
    except FileNotFoundError:
        print(f"❌ Test file '{test_file}' not found")
        print("Create a test audio file or use a valid path to test")
