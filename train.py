"""
Training Script for AI Voice Detection Model
ENHANCED with:
- Audio chunking for long files (splits into 20 sec segments)
- V3 features (200 features: phase, jitter, pause patterns, etc.)
- Ensemble learning for better generalization
"""

import os
import sys
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings

# Use V3 features with advanced analysis (phase, jitter, pause patterns)
from features_v3 import extract_features

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Supported audio formats
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# Languages we support - train on all available
LANGUAGES = ['english', 'tamil', 'telugu', 'hindi', 'malayalam']

# Chunking settings for long audio files
CHUNK_DURATION = 20  # seconds per chunk
MIN_CHUNK_DURATION = 10  # minimum usable chunk length
SAMPLE_RATE = 22050


def get_audio_duration(file_path: Path) -> float:
    """Get audio file duration in seconds."""
    try:
        return librosa.get_duration(path=str(file_path))
    except:
        return 0


def extract_features_from_chunk(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract features from an audio chunk (numpy array)."""
    # Save chunk to temp file and extract features
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y, sr)
    
    try:
        features = extract_features(tmp_path)
        return features
    finally:
        os.unlink(tmp_path)


def process_audio_file(file_path: Path, label: int) -> list:
    """
    Process an audio file, splitting into chunks if long.
    Returns list of (features, label, chunk_name) tuples
    """
    results = []
    
    try:
        duration = get_audio_duration(file_path)
        
        if duration <= CHUNK_DURATION + 5:
            # Short file - process directly
            features = extract_features(str(file_path))
            results.append((features, label, f"{file_path.name}"))
        else:
            # Long file - split into chunks
            y, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
            
            chunk_samples = int(CHUNK_DURATION * sr)
            min_chunk_samples = int(MIN_CHUNK_DURATION * sr)
            
            chunk_idx = 0
            start = 0
            
            while start < len(y):
                end = start + chunk_samples
                
                # Last chunk handling
                if end > len(y):
                    remaining = len(y) - start
                    if remaining >= min_chunk_samples:
                        chunk = y[start:]
                    else:
                        break  # Skip too-short final chunk
                else:
                    chunk = y[start:end]
                
                # Extract features from chunk
                try:
                    features = extract_features_from_chunk(chunk, sr)
                    chunk_name = f"{file_path.stem}_chunk{chunk_idx}"
                    results.append((features, label, chunk_name))
                    chunk_idx += 1
                except:
                    pass  # Skip failed chunks
                
                start = end
            
    except Exception as e:
        raise e
    
    return results


def collect_audio_files(data_dir: Path) -> tuple:
    """
    Collect all audio files from the data directory structure.
    
    Expected structure:
    data/
    ├── ai/
    │   ├── tamil/, telugu/, malayalam/, english/, hindi/
    └── human/
        ├── tamil/, telugu/, malayalam/, english/, hindi/
    
    Returns:
        tuple: (file_paths, labels, languages)
    """
    file_paths = []
    labels = []  # 0 = Human, 1 = AI
    file_languages = []
    
    for category in ['human', 'ai']:
        category_dir = data_dir / category
        label = 1 if category == 'ai' else 0
        
        if not category_dir.exists():
            print(f"Warning: {category_dir} does not exist")
            continue
        
        for language in LANGUAGES:
            lang_dir = category_dir / language
            
            if not lang_dir.exists():
                continue
            
            for file_path in lang_dir.iterdir():
                if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    file_paths.append(file_path)
                    labels.append(label)
                    file_languages.append(language)
    
    return file_paths, labels, file_languages


def extract_all_features(file_paths: list, labels: list) -> tuple:
    """
    Extract features from all audio files, chunking long files.
    """
    X = []
    y = []
    chunk_names = []
    
    total = len(file_paths)
    print(f"\nProcessing {total} audio files (long files will be chunked)...")
    print(f"Chunk settings: {CHUNK_DURATION}s chunks, min {MIN_CHUNK_DURATION}s")
    print("-" * 60)
    
    total_chunks = 0
    
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            duration = get_audio_duration(file_path)
            label_str = "AI" if label == 1 else "Human"
            
            if duration > CHUNK_DURATION + 5:
                num_chunks = int(duration // CHUNK_DURATION)
                print(f"  [{i+1}/{total}] {file_path.name} ({duration:.0f}s) → ~{num_chunks} chunks [{label_str}]...", end=" ", flush=True)
            else:
                print(f"  [{i+1}/{total}] {file_path.name} ({duration:.0f}s) [{label_str}]...", end=" ", flush=True)
            
            results = process_audio_file(file_path, label)
            
            for features, lbl, name in results:
                X.append(features)
                y.append(lbl)
                chunk_names.append(name)
                total_chunks += 1
            
            if len(results) > 1:
                print(f"✓ ({len(results)} chunks)")
            else:
                print("✓")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print("-" * 60)
    print(f"Total training samples (after chunking): {total_chunks}")
    
    return np.array(X), np.array(y), chunk_names


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train an ensemble classifier with calibration for better generalization.
    Uses combination of Random Forest + Gradient Boosting for robustness.
    
    Returns:
        tuple: (trained model, scaler, metrics dict)
    """
    print("\n" + "="*50)
    print("TRAINING ENSEMBLE MODEL")
    print("="*50)
    
    # Use RobustScaler - less sensitive to outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Replace any NaN/Inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle small datasets
    if len(X) < 10:
        print(f"\nSmall dataset mode: {len(X)} samples")
        print("Training on all data (no train/test split)")
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
        test_size_used = 0
    else:
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        test_size_used = len(X_test)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {test_size_used} samples")
    print(f"AI samples: {sum(y)}, Human samples: {len(y) - sum(y)}")
    
    # ============ ENSEMBLE MODEL ============
    print("\nTraining ensemble classifier...")
    
    # Model 1: Random Forest - good at capturing non-linear patterns
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',  # Handle any class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: Gradient Boosting - good at finding subtle patterns
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # Combine with soft voting (uses probabilities)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1, 1]  # Equal weight
    )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    
    # Calibrate probabilities for better confidence scores
    if len(X_train) >= 20:
        print("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
        calibrated_model.fit(X_train, y_train)
        model = calibrated_model
    else:
        model = ensemble
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    # Check for overfitting
    if train_accuracy - test_accuracy > 0.15:
        print("\n⚠️  Warning: Possible overfitting detected!")
        print(f"   Train-Test gap: {(train_accuracy - test_accuracy)*100:.1f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Human correctly identified: {cm[0][0]}")
    print(f"  Human misclassified as AI: {cm[0][1]}")
    print(f"  AI correctly identified: {cm[1][1]}")
    print(f"  AI misclassified as Human: {cm[1][0]}")
    
    # Cross-validation with stratified folds
    if len(y) >= 10:
        print("\nStratified Cross-validation (5-fold):")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='f1_weighted')
        print(f"  F1 Scores: {[f'{s:.2%}' for s in cv_scores]}")
        print(f"  Mean F1: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    else:
        print("\nSkipping cross-validation (need at least 10 samples)")
        cv_mean, cv_std = test_accuracy, 0.0
    
    # Feature importance from Random Forest
    print("\n" + "="*50)
    print("TOP 15 IMPORTANT FEATURES")
    print("="*50)
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    # Get feature names from v3
    try:
        from features_v3 import FEATURE_NAMES
        feature_names = FEATURE_NAMES
    except:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    for i, idx in enumerate(indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "f1_score": float(f1),
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "total_samples": len(y),
        "ai_samples": int(sum(y)),
        "human_samples": int(len(y) - sum(y)),
        "model_type": "ensemble_rf_gb_calibrated"
    }
    
    return model, scaler, metrics


def get_feature_names():
    """Return descriptive names for all 125 features."""
    names = []
    
    # Part 1: Basic Spectral (40)
    for i in range(13):
        names.append(f"mfcc_{i+1}_mean")
    for i in range(13):
        names.append(f"mfcc_{i+1}_std")
    for i in range(13):
        names.append(f"delta_mfcc_{i+1}_mean")
    names.append("spectral_centroid_mean")
    
    # Part 2: Pitch Analysis (25)
    names.extend([
        "pitch_mean", "pitch_std", "pitch_range", "pitch_median",
        "jitter_mean", "jitter_std", "jitter_relative",
        "pitch_p25", "pitch_p75", "pitch_iqr", "pitch_roughness",
        "pitch_skew", "pitch_kurtosis", "pitch_accel_mean", "pitch_accel_std",
        "voiced_ratio", "voiced_seg_mean", "voiced_seg_std", "voiced_seg_max",
        "pitch_entropy", "vibrato_indicator", "local_pitch_std_mean", "local_pitch_std_std",
        "pitch_pad1", "pitch_pad2"
    ])
    
    # Part 3: Energy/Pause (20)
    names.extend([
        "rms_mean", "rms_std", "dynamic_range",
        "num_pauses", "pause_mean", "pause_std", "pause_max", "silence_ratio",
        "energy_diff_mean", "energy_diff_std", "energy_entropy", "speech_rate",
        "local_energy_var_mean", "local_energy_var_std",
        "energy_pad1", "energy_pad2", "energy_pad3", "energy_pad4", "energy_pad5", "energy_pad6"
    ])
    
    # Part 4: Temporal Smoothness (15)
    names.extend([
        "spectral_flux_mean", "spectral_flux_std", "spectral_flux_max",
        "flux_diff_mean", "flux_diff_std", "mfcc_diff_mean", "mfcc_diff_std",
        "temporal_flatness", "flux_skew", "flux_kurtosis",
        "temporal_pad1", "temporal_pad2", "temporal_pad3", "temporal_pad4", "temporal_pad5"
    ])
    
    # Part 5: Shimmer (15)
    names.extend([
        "shimmer", "shimmer_std", "apq",
        "zc_interval_mean", "zc_interval_std", "zc_interval_cv",
        "hnr", "noise_floor", "harmonic_std",
        "shimmer_pad1", "shimmer_pad2", "shimmer_pad3", "shimmer_pad4", "shimmer_pad5", "shimmer_pad6"
    ])
    
    # Part 6: Spectral Texture (10)
    names.extend([
        "spectral_flatness_mean", "spectral_flatness_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_contrast_mean", "spectral_contrast_std",
        "texture_pad1", "texture_pad2"
    ])
    
    return names


def save_model(model, scaler, metrics: dict):
    """Save the trained model and scaler to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_path = MODELS_DIR / "voice_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    metrics_path = MODELS_DIR / "metrics.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(metrics, metrics_path)
    
    print("\n" + "="*50)
    print("MODEL SAVED")
    print("="*50)
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Metrics: {metrics_path}")


def main():
    """Main training pipeline."""
    print("="*50)
    print("AI VOICE DETECTION - MODEL TRAINING")
    print("="*50)
    print(f"\nUsing V3 Features: 200 advanced features")
    print(f"  - Phase coherence & synthesis artifacts")
    print(f"  - Enhanced jitter/shimmer (RAP, PPQ, APQ)")
    print(f"  - Pause patterns & breathing detection")
    print(f"  - Formant transitions & prosody")
    print(f"  - Filler word patterns (uh, umm energy signatures)")
    print(f"\nAudio Chunking: {CHUNK_DURATION}s chunks for long files")
    
    # Check data directory
    if not DATA_DIR.exists():
        print(f"\nError: Data directory not found: {DATA_DIR}")
        print("Please create the following structure:")
        print("  data/ai/<language>/*.wav")
        print("  data/human/<language>/*.wav")
        sys.exit(1)
    
    # Collect audio files
    print("\nCollecting audio files...")
    file_paths, labels, languages = collect_audio_files(DATA_DIR)
    
    if len(file_paths) == 0:
        print("\nError: No audio files found!")
        print("\nPlease add audio samples to:")
        for lang in LANGUAGES:
            print(f"  - data/ai/{lang}/")
            print(f"  - data/human/{lang}/")
        print("\nSupported formats: WAV, MP3, FLAC, OGG, M4A")
        sys.exit(1)
    
    print(f"\nFound {len(file_paths)} audio files:")
    print(f"  - AI voices: {labels.count(1)}")
    print(f"  - Human voices: {labels.count(0)}")
    
    # Check minimum samples
    if len(file_paths) < 10:
        print("\nWarning: Very few samples. Recommend at least 20 samples per category.")
    
    if labels.count(0) == 0 or labels.count(1) == 0:
        print("\nError: Need samples from both AI and Human categories!")
        sys.exit(1)
    
    # Extract features
    X, y, successful_files = extract_all_features(file_paths, labels)
    
    if len(X) < 2:
        print("\nError: Not enough successfully processed samples for training!")
        print("Need at least 2 samples (1 AI + 1 Human).")
        sys.exit(1)
    
    # Train model
    model, scaler, metrics = train_model(X, y)
    
    # Save model
    save_model(model, scaler, metrics)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\nYou can now use the API to detect AI voices.")
    print("Run: python -m uvicorn main:app --reload")


if __name__ == "__main__":
    main()
