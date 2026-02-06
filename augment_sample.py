"""
Create augmented versions of sample AI voice to help model generalize
to advanced AI voice generators (ElevenLabs-style voices)
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLE_PATH = "data/sample voice 1.mp3"
OUTPUT_DIR = Path("data/ai/english")

def augment_audio(y, sr):
    """Create variations of the audio"""
    augmented = []
    
    # Original
    augmented.append(("orig", y))
    
    # Pitch shifts
    augmented.append(("pitch_up1", librosa.effects.pitch_shift(y, sr=sr, n_steps=1)))
    augmented.append(("pitch_up2", librosa.effects.pitch_shift(y, sr=sr, n_steps=2)))
    augmented.append(("pitch_down1", librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)))
    augmented.append(("pitch_down2", librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)))
    
    # Time stretch (speed changes)
    augmented.append(("faster", librosa.effects.time_stretch(y, rate=1.1)))
    augmented.append(("slower", librosa.effects.time_stretch(y, rate=0.9)))
    
    # Add slight noise
    noise = np.random.normal(0, 0.005, len(y))
    augmented.append(("noisy1", y + noise))
    
    noise2 = np.random.normal(0, 0.01, len(y))
    augmented.append(("noisy2", y + noise2))
    
    # Volume variations
    augmented.append(("louder", y * 1.2))
    augmented.append(("quieter", y * 0.8))
    
    return augmented

def create_chunks(y, sr, chunk_duration=5):
    """Split audio into chunks"""
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    for i in range(0, len(y) - chunk_samples, chunk_samples // 2):  # 50% overlap
        chunk = y[i:i + chunk_samples]
        if len(chunk) == chunk_samples:
            chunks.append(chunk)
    return chunks

def main():
    print("=" * 50)
    print("AUGMENTING SAMPLE AI VOICE")
    print("=" * 50)
    
    # Load sample
    print(f"\nLoading: {SAMPLE_PATH}")
    y, sr = librosa.load(SAMPLE_PATH, sr=22050)
    print(f"Duration: {len(y)/sr:.2f}s, Sample rate: {sr}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    # Create augmented versions
    print("\nCreating augmented versions...")
    augmented = augment_audio(y, sr)
    
    for name, audio in augmented:
        # Save full augmented version
        output_path = OUTPUT_DIR / f"advanced_ai_{name}.wav"
        sf.write(str(output_path), audio, sr)
        print(f"  Created: {output_path.name}")
        count += 1
        
        # Also create chunks from each augmented version
        chunks = create_chunks(audio, sr, chunk_duration=5)
        for ci, chunk in enumerate(chunks[:3]):  # Take first 3 chunks
            chunk_path = OUTPUT_DIR / f"advanced_ai_{name}_chunk{ci}.wav"
            sf.write(str(chunk_path), chunk, sr)
            count += 1
    
    print(f"\n✓ Created {count} augmented AI voice samples")
    print("\nNow retrain: python train.py")

if __name__ == "__main__":
    main()
