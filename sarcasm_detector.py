"""
Sarcasm Detection Module
Detects sarcasm using acoustic cues: pitch, energy, tempo, and emotion-tone mismatch
"""

import numpy as np
import librosa

class SarcasmDetector:
    """
    Sarcasm Detection System using acoustic analysis
    
    Detection based on:
    1. Exaggerated pitch variation
    2. Irregular energy patterns
    3. Unusual speaking tempo
    4. Emotion-tone mismatch
    """
    
    def __init__(self, threshold=0.6):
        """
        Initialize sarcasm detector
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for sarcasm detection (0.0 to 1.0)
        """
        self.threshold = threshold
        print(f"🎭 Sarcasm Detector initialized (threshold: {threshold})")
    
    def analyze_pitch_variation(self, audio, sr=22050):
        """
        Analyze pitch variation in speech
        
        Sarcastic speech characteristics:
        - High pitch standard deviation (exaggerated intonation)
        - Large pitch range (dramatic ups and downs)
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        
        Returns:
        --------
        pitch_std : float
            Standard deviation of pitch
        pitch_range : float
            Range of pitch values
        """
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        pitch_values = []
        
        # Get pitch at each time frame
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            
            if pitch > 0:  # Only consider valid pitches
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_std = np.std(pitch_values)
            pitch_range = np.max(pitch_values) - np.min(pitch_values)
            return pitch_std, pitch_range
        
        return 0, 0
    
    def analyze_energy_pattern(self, audio):
        """
        Analyze energy/loudness pattern
        
        Sarcastic speech indicators:
        - High energy variance (inconsistent volume)
        - Many sudden peaks (emphasis on certain words)
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        
        Returns:
        --------
        energy_variance : float
            Variance in energy
        energy_peaks : int
            Number of energy peaks
        """
        # Calculate RMS (Root Mean Square) energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Energy variance
        energy_variance = np.var(rms)
        
        # Count peaks in energy
        energy_peaks = len(librosa.util.peak_pick(
            rms, 
            pre_max=3, 
            post_max=3, 
            pre_avg=3, 
            post_avg=5, 
            delta=0.5, 
            wait=10
        ))
        
        return energy_variance, energy_peaks
    
    def analyze_tempo(self, audio, sr=22050):
        """
        Analyze speaking tempo
        
        Sarcastic speech patterns:
        - Slower than normal (< 80 BPM)
        - Faster than normal (> 140 BPM)
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        
        Returns:
        --------
        tempo : float
            Estimated tempo in BPM
        """
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return tempo
    
    def detect_sarcasm(self, audio, sr=22050, predicted_emotion="neutral"):
        """
        Main sarcasm detection algorithm
        
        Combines multiple acoustic cues to determine sarcasm probability
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        predicted_emotion : str
            Emotion predicted by emotion detection model
        
        Returns:
        --------
        result : dict
            Dictionary containing:
            - is_sarcastic: bool
            - probability: float
            - confidence: float (percentage)
            - indicators: list of detected indicators
            - acoustic_features: dict of feature values
        """
        
        # Extract acoustic features
        pitch_std, pitch_range = self.analyze_pitch_variation(audio, sr)
        energy_var, energy_peaks = self.analyze_energy_pattern(audio)
        tempo = self.analyze_tempo(audio, sr)
        
        # Initialize sarcasm score and indicators
        sarcasm_score = 0
        indicators = []
        
        # Indicator 1: Exaggerated pitch variation
        if pitch_std > 50:
            sarcasm_score += 0.25
            indicators.append("Exaggerated pitch variation")
        
        # Indicator 2: Irregular energy pattern
        if energy_var > 0.01:
            sarcasm_score += 0.2
            indicators.append("Irregular energy pattern")
        
        # Indicator 3: Unusual tempo
        if tempo < 80 or tempo > 140:
            sarcasm_score += 0.15
            indicators.append("Unusual speaking tempo")
        
        # Indicator 4: Emotion-tone mismatch (positive emotion with high pitch variation)
        if predicted_emotion in ['happy', 'surprised'] and pitch_std > 60:
            sarcasm_score += 0.3
            indicators.append("Emotion-tone mismatch (exaggerated positivity)")
        
        # Indicator 5: Exaggerated negative emotion
        if predicted_emotion in ['sad', 'angry'] and pitch_range > 150:
            sarcasm_score += 0.2
            indicators.append("Exaggerated negative emotion")
        
        # Calculate final probability
        sarcasm_probability = min(sarcasm_score, 1.0)
        is_sarcastic = sarcasm_probability >= self.threshold
        
        # Return comprehensive results
        return {
            'is_sarcastic': is_sarcastic,
            'probability': sarcasm_probability,
            'confidence': sarcasm_probability * 100,
            'indicators': indicators,
            'acoustic_features': {
                'pitch_std': pitch_std,
                'pitch_range': pitch_range,
                'energy_variance': energy_var,
                'energy_peaks': energy_peaks,
                'tempo': tempo
            }
        }


# Test the sarcasm detector
if __name__ == "__main__":
    print("Testing Sarcasm Detector...")
    print("=" * 50)
    
    # You'll need a test audio file to run this
    test_file = "D://Final Project//Audio_Song_Actors_01-24//Actor_01//03-02-01-01-01-01-01.wav"

    try:
        audio, sr = librosa.load(test_file, sr=22050)
        
        detector = SarcasmDetector(threshold=0.6)
        result = detector.detect_sarcasm(audio, sr, predicted_emotion="happy")
        
        print("\n✅ Sarcasm Detection Results:")
        print(f"Is Sarcastic: {result['is_sarcastic']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Indicators: {result['indicators']}")
        print(f"Acoustic Features: {result['acoustic_features']}")
        
    except FileNotFoundError:
        print(f"❌ Test file '{test_file}' not found")
        print("Add a valid audio file to test the sarcasm detector")
