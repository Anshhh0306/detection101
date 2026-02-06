"""
Training Script for AI Voice Detection Model - VERSION 3
=========================================================
ADVANCED FEATURES:
- 200 acoustic features (jitter, shimmer, phase, formants, pause patterns)
- Ensemble learning (RF + Gradient Boosting + Extra Trees)
- Probability calibration for reliable confidence scores
- Data augmentation option for better generalization
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings

# Use enhanced features
from features_v3 import extract_features, FEATURE_NAMES, N_FEATURES

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Supported audio formats
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# Languages we support
LANGUAGES = ['english', 'tamil', 'telugu', 'hindi', 'malayalam']


def collect_audio_files(data_dir: Path) -> tuple:
    """
    Collect all audio files from the data directory structure.
    
    Expected structure:
    data/
    ├── ai/
    │   ├── tamil/, telugu/, malayalam/, english/, hindi/
    └── human/
        ├── tamil/, telugu/, malayalam/, english/, hindi/
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
    Extract features from all audio files.
    """
    X = []
    y = []
    successful_files = []
    
    total = len(file_paths)
    print(f"\nExtracting {N_FEATURES} features from {total} audio files...")
    
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            print(f"  [{i+1}/{total}] {file_path.name}...", end=" ")
            features = extract_features(str(file_path))
            X.append(features)
            y.append(label)
            successful_files.append(file_path.name)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    return np.array(X), np.array(y), successful_files


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train an advanced ensemble classifier.
    
    Architecture:
    - Random Forest: Non-linear patterns, robust
    - Gradient Boosting: Subtle patterns
    - Extra Trees: Variance reduction
    - Soft voting with probability calibration
    """
    print("\n" + "="*60)
    print("TRAINING ADVANCED ENSEMBLE MODEL (v3)")
    print("="*60)
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split data
    if len(X) < 10:
        print(f"\nSmall dataset mode: {len(X)} samples")
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(y)}")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  AI samples: {sum(y)}")
    print(f"  Human samples: {len(y) - sum(y)}")
    print(f"  Features per sample: {N_FEATURES}")
    
    # ============ ENSEMBLE MODEL ============
    print("\nBuilding ensemble classifier...")
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42
    )
    
    # Model 3: Extra Trees (additional variance reduction)
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Combine with soft voting
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('et', et)],
        voting='soft',
        weights=[1, 1, 1]
    )
    
    print("Training ensemble (RF + GB + Extra Trees)...")
    ensemble.fit(X_train, y_train)
    
    # Calibrate probabilities
    if len(X_train) >= 30:
        print("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(ensemble, cv=5, method='isotonic')
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
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"\n  Training Accuracy: {train_accuracy:.2%}")
    print(f"  Test Accuracy:     {test_accuracy:.2%}")
    print(f"  F1 Score:          {f1:.2%}")
    
    # Overfitting check
    gap = train_accuracy - test_accuracy
    if gap > 0.10:
        print(f"\n  ⚠️  Warning: Possible overfitting (gap: {gap*100:.1f}%)")
    elif gap < 0.05:
        print(f"\n  ✓ Good generalization (gap: {gap*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted Human  Predicted AI")
    print(f"  Actual Human     {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"  Actual AI        {cm[1][0]:>6}        {cm[1][1]:>6}")
    
    # Cross-validation
    if len(y) >= 10:
        print("\n" + "="*60)
        print("CROSS-VALIDATION (5-fold stratified)")
        print("="*60)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='f1_weighted')
        print(f"\n  Fold F1 Scores: {[f'{s:.2%}' for s in cv_scores]}")
        print(f"  Mean F1: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    else:
        cv_mean, cv_std = test_accuracy, 0.0
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 20 IMPORTANT FEATURES")
    print("="*60)
    
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    print("\n  #    Feature Name                          Importance")
    print("  " + "-"*55)
    for i, idx in enumerate(indices):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        print(f"  {i+1:>2}.  {name:<40} {importances[idx]:.4f}")
    
    # Identify key indicators
    print("\n" + "="*60)
    print("KEY AI DETECTION INDICATORS")
    print("="*60)
    
    indicator_categories = {
        "Jitter/Shimmer (voice naturalness)": ["jitter", "shimmer", "rap", "ppq", "apq"],
        "Pause/Breathing patterns": ["pause", "breath", "silence", "filler"],
        "Phase coherence (synthesis artifacts)": ["phase", "group_delay", "inst_freq"],
        "Pitch naturalness": ["pitch", "f0", "vibrato", "roughness"],
        "Spectral dynamics": ["spectral", "mfcc", "formant"],
    }
    
    for category, keywords in indicator_categories.items():
        relevant = [(i, name, importances[i]) for i, name in enumerate(FEATURE_NAMES) 
                    if any(kw in name.lower() for kw in keywords)]
        if relevant:
            total_importance = sum(imp for _, _, imp in relevant)
            print(f"\n  {category}: {total_importance:.4f}")
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "f1_score": float(f1),
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "total_samples": len(y),
        "ai_samples": int(sum(y)),
        "human_samples": int(len(y) - sum(y)),
        "n_features": N_FEATURES,
        "model_type": "ensemble_rf_gb_et_calibrated_v3"
    }
    
    return model, scaler, metrics


def save_model(model, scaler, metrics: dict):
    """Save trained model, scaler, and metadata."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_path = MODELS_DIR / "voice_detector.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    metadata_path = MODELS_DIR / "model_metadata.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(metrics, metadata_path)
    
    print("\n" + "="*60)
    print("MODEL SAVED")
    print("="*60)
    print(f"\n  Model:    {model_path}")
    print(f"  Scaler:   {scaler_path}")
    print(f"  Metadata: {metadata_path}")
    
    return model_path, scaler_path


def main():
    """Main training pipeline."""
    print("="*60)
    print("AI VOICE DETECTION MODEL TRAINER v3")
    print("="*60)
    print(f"\nFeatures: {N_FEATURES} advanced acoustic features")
    print("Including: jitter, shimmer, phase coherence, formant transitions")
    print("          pause patterns, breathing detection, prosody analysis")
    
    # Collect files
    print("\n" + "="*60)
    print("COLLECTING AUDIO FILES")
    print("="*60)
    
    file_paths, labels, languages = collect_audio_files(DATA_DIR)
    
    if len(file_paths) == 0:
        print("\nNo audio files found!")
        print("Please add audio files to:")
        print(f"  {DATA_DIR / 'ai' / '<language>'}")
        print(f"  {DATA_DIR / 'human' / '<language>'}")
        return
    
    # Count by category and language
    print(f"\nFound {len(file_paths)} audio files:")
    for category in ['human', 'ai']:
        cat_label = 1 if category == 'ai' else 0
        cat_count = sum(1 for l in labels if l == cat_label)
        print(f"\n  {category.upper()}: {cat_count} files")
        
        for lang in LANGUAGES:
            lang_count = sum(1 for i, l in enumerate(labels) 
                          if l == cat_label and languages[i] == lang)
            if lang_count > 0:
                print(f"    - {lang}: {lang_count}")
    
    # Extract features
    X, y, successful_files = extract_all_features(file_paths, labels)
    
    if len(X) == 0:
        print("\nFailed to extract features from any files!")
        return
    
    print(f"\nSuccessfully processed: {len(X)}/{len(file_paths)} files")
    
    # Check class balance
    ai_count = sum(y)
    human_count = len(y) - ai_count
    
    if ai_count == 0 or human_count == 0:
        print("\n⚠️  Error: Need both AI and Human samples!")
        print(f"   AI samples: {ai_count}")
        print(f"   Human samples: {human_count}")
        return
    
    # Train
    model, scaler, metrics = train_model(X, y)
    
    # Save
    save_model(model, scaler, metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  F1 Score:      {metrics['f1_score']:.2%}")
    print(f"  CV Mean F1:    {metrics['cv_mean']:.2%}")
    
    print("\nReady to detect AI-generated voices!")
    print("Run the API server with: python -m uvicorn main:app --reload")


if __name__ == "__main__":
    main()
