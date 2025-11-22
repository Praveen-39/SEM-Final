"""
CNN-LSTM Model Training Module
Trains deep learning model for speech emotion recognition
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import librosa
from feature_extraction import extract_features

# Emotion labels (RAVDESS dataset)
EMOTIONS = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']


def load_ravdess_data(data_path="D://Final Project//Audio_Song_Actors_01-24"):
    """
    Load and process RAVDESS dataset
    
    RAVDESS file naming format:
    03-01-06-01-02-01-12.wav
    
    Position 3 (06) = Emotion:
    01 = neutral, 02 = calm, 03 = happy, 04 = sad
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    
    Parameters:
    -----------
    data_path : str
        Path to RAVDESS dataset folder
    
    Returns:
    --------
    features : numpy.ndarray
        Array of audio features
    labels : numpy.ndarray
        Array of emotion labels
    """
    features = []
    labels = []
    
    print("=" * 60)
    print("LOADING RAVDESS DATASET")
    print("=" * 60)
    
    file_count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                try:
                    # Parse emotion from filename
                    parts = file.split('-')
                    emotion_code = int(parts[2])  # 3rd position is emotion
                    
                    # Map code to emotion name
                    emotion = EMOTIONS[emotion_code - 1]
                    
                    # Get full file path
                    file_path = os.path.join(root, file)
                    
                    # Extract features
                    feature = extract_features(file_path)
                    
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)
                        file_count += 1
                        
                        if file_count % 100 == 0:
                            print(f"Processed {file_count} files...")
                
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
    
    print(f"\n✅ Total samples loaded: {len(features)}")
    print(f"Unique emotions: {set(labels)}")
    
    return np.array(features), np.array(labels)


def build_cnn_lstm_model(input_shape, num_classes):
    """
    Build CNN-LSTM hybrid model for emotion recognition
    
    Architecture:
    - CNN layers: Extract spatial features from audio
    - LSTM layers: Learn temporal patterns
    - Dense layers: Classification
    
    Parameters:
    -----------
    input_shape : int
        Number of features per sample
    num_classes : int
        Number of emotion categories
    
    Returns:
    --------
    model : keras.Model
        Compiled CNN-LSTM model
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_shape, 1)),
        
        # CNN Block 1
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # CNN Block 2
        layers.Conv1D(256, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # CNN Block 3
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # LSTM Block 1
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        
        # LSTM Block 2
        layers.LSTM(64),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_emotion_model(data_path="D://Final Project//Audio_Song_Actors_01-24"):
    """
    Main training function
    
    Parameters:
    -----------
    data_path : str
        Path to RAVDESS dataset
    
    Returns:
    --------
    model : keras.Model
        Trained model
    label_encoder : LabelEncoder
        Label encoder
    history : keras.History
        Training history
    """
    
    print("\n" + "=" * 60)
    print("SPEECH EMOTION RECOGNITION - MODEL TRAINING")
    print("CNN-LSTM Architecture")
    print("=" * 60 + "\n")
    
    # Step 1: Load data
    X, y = load_ravdess_data(data_path)
    
    if len(X) == 0:
        print("\n❌ ERROR: No data loaded!")
        print("Make sure RAVDESS dataset is in 'data/RAVDESS/' folder")
        return None, None, None
    
    # Step 2: Encode labels
    print("\n" + "=" * 60)
    print("ENCODING LABELS")
    print("=" * 60)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Step 3: Reshape data for CNN-LSTM
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y_categorical.shape}")
    
    # Step 4: Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_categorical
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")
    
    # Step 5: Build model
    print("\n" + "=" * 60)
    print("BUILDING CNN-LSTM MODEL")
    print("=" * 60)
    model = build_cnn_lstm_model(X.shape[1], len(EMOTIONS))
    model.summary()
    
    # Step 6: Setup callbacks
    print("\n" + "=" * 60)
    print("SETTING UP TRAINING CALLBACKS")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/emotion_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Step 7: Train model
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print("This may take 20-30 minutes...\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 8: Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # Step 9: Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model.save('models/emotion_model.h5')
    np.save('models/label_encoder.npy', label_encoder.classes_)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: models/emotion_model.h5")
    print(f"📁 Label encoder saved to: models/label_encoder.npy")
    
    return model, label_encoder, history


# Main execution
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Speech Emotion Detection with Sarcasm Detection        ║
    ║  CNN-LSTM Model Training                                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Train model
    model, label_encoder, history = train_emotion_model()
    
    if model is not None:
        print("\n✅ Model training successful!")
        print("You can now run: streamlit run app.py")
    else:
        print("\n❌ Model training failed!")
        print("Check if RAVDESS dataset is in 'data/RAVDESS/' folder")
