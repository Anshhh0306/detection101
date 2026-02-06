"""
Test script for AI Voice Detection
Tests the model with sample AI and Human audio files
"""

from model import predict_voice
import base64
import os

def test_direct_prediction():
    """Test direct model prediction (not via API)"""
    print("=" * 60)
    print("TESTING DIRECT MODEL PREDICTION")
    print("=" * 60)
    
    # Test AI sample
    ai_sample = 'data/ai/english/ai_sample_1.mp3'
    if os.path.exists(ai_sample):
        result = predict_voice(ai_sample, 'English')
        print(f"\n[AI Sample] {ai_sample}")
        print(f"  Classification: {result['classification']}")
        print(f"  Confidence: {result['confidenceScore']}")
        print(f"  Explanation: {result['explanation']}")
    
    # Test Human sample
    human_sample = 'data/human/english/LJ001-0002.wav'
    if os.path.exists(human_sample):
        result2 = predict_voice(human_sample, 'English')
        print(f"\n[Human Sample] {human_sample}")
        print(f"  Classification: {result2['classification']}")
        print(f"  Confidence: {result2['confidenceScore']}")
        print(f"  Explanation: {result2['explanation']}")


def test_api_format():
    """Test the API request/response format"""
    print("\n" + "=" * 60)
    print("TESTING API FORMAT")
    print("=" * 60)
    
    # Simulate what the API receives
    sample_file = 'data/ai/english/ai_sample_1.mp3'
    if os.path.exists(sample_file):
        # Read and encode file as base64 (simulating API input)
        with open(sample_file, 'rb') as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"\nSimulated API Request:")
        print(f"  language: English")
        print(f"  audioFormat: mp3")
        print(f"  audioBase64: {audio_base64[:50]}... ({len(audio_base64)} chars)")
        
        # Decode back and test
        decoded = base64.b64decode(audio_base64)
        
        # Save to temp and test
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        
        result = predict_voice(tmp_path, 'English')
        os.unlink(tmp_path)
        
        print(f"\nSimulated API Response:")
        print(f"  status: success")
        print(f"  language: English")
        print(f"  classification: {result['classification']}")
        print(f"  confidenceScore: {result['confidenceScore']}")
        print(f"  explanation: {result['explanation']}")


if __name__ == "__main__":
    test_direct_prediction()
    test_api_format()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
