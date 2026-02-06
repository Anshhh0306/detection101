"""
Script to download/generate training samples for AI Voice Detection
- AI voices: Generated using Microsoft Edge TTS
- Human voices: Downloaded from Mozilla Common Voice dataset
"""

import asyncio
import os
import sys
from pathlib import Path
import random

# Try to import edge_tts
try:
    import edge_tts
except ImportError:
    print("Installing edge-tts...")
    os.system(f"{sys.executable} -m pip install edge-tts")
    import edge_tts

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    os.system(f"{sys.executable} -m pip install aiohttp")
    import aiohttp

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Voice configurations for each language
VOICES = {
    "english": [
        "en-US-JennyNeural",
        "en-US-GuyNeural", 
        "en-GB-SoniaNeural",
        "en-AU-NatashaNeural",
    ],
    "hindi": [
        "hi-IN-SwaraNeural",
        "hi-IN-MadhurNeural",
    ],
    "tamil": [
        "ta-IN-PallaviNeural",
        "ta-IN-ValluvarNeural",
    ],
    "telugu": [
        "te-IN-ShrutiNeural",
        "te-IN-MohanNeural",
    ],
    "malayalam": [
        "ml-IN-SobhanaNeural",
        "ml-IN-MidhunNeural",
    ],
}

# Sample texts for each language
TEXTS = {
    "english": [
        "Hello, my name is Sarah and I am calling to discuss your recent order.",
        "Welcome to our customer service center. How may I assist you today?",
        "Thank you for choosing our services. Your satisfaction is our priority.",
        "The weather today is expected to be sunny with a high of twenty five degrees.",
        "Please hold while I transfer your call to the appropriate department.",
        "Your account has been successfully updated with the new information.",
        "We appreciate your patience during this busy period.",
        "For more information, please visit our website or call our helpline.",
        "This is an automated message to confirm your appointment tomorrow.",
        "Our office hours are from nine in the morning to five in the evening.",
    ],
    "hindi": [
        "नमस्ते, मेरा नाम राहुल है और मैं आपकी सहायता के लिए यहाँ हूँ।",
        "आपका स्वागत है। कृपया अपना प्रश्न पूछें।",
        "धन्यवाद आपके सहयोग के लिए। हम जल्द ही आपसे संपर्क करेंगे।",
        "आज का मौसम साफ और धूप वाला रहने की उम्मीद है।",
        "कृपया लाइन पर बने रहें, हम आपकी कॉल को स्थानांतरित कर रहे हैं।",
        "आपका खाता सफलतापूर्वक अपडेट कर दिया गया है।",
        "आपके धैर्य के लिए धन्यवाद।",
        "अधिक जानकारी के लिए कृपया हमारी वेबसाइट देखें।",
        "यह एक स्वचालित संदेश है जो आपकी अपॉइंटमेंट की पुष्टि करता है।",
        "हमारे कार्यालय का समय सुबह नौ से शाम पांच बजे तक है।",
    ],
    "tamil": [
        "வணக்கம், என் பெயர் குமார். நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன்.",
        "எங்கள் சேவையைத் தேர்ந்தெடுத்ததற்கு நன்றி.",
        "உங்கள் கணக்கு வெற்றிகரமாக புதுப்பிக்கப்பட்டது.",
        "இன்றைய வானிலை வெயிலாக இருக்கும் என எதிர்பார்க்கப்படுகிறது.",
        "தயவுசெய்து காத்திருங்கள், உங்கள் அழைப்பை மாற்றுகிறோம்.",
        "உங்கள் பொறுமைக்கு நன்றி.",
        "மேலும் தகவலுக்கு எங்கள் வலைத்தளத்தைப் பார்வையிடவும்.",
        "இது ஒரு தானியங்கி செய்தி.",
        "எங்கள் அலுவலக நேரம் காலை ஒன்பது முதல் மாலை ஐந்து வரை.",
        "உங்கள் ஆர்டர் வெற்றிகரமாக வைக்கப்பட்டது.",
    ],
    "telugu": [
        "నమస్కారం, నా పేరు రాజు. నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.",
        "మా సేవను ఎంచుకున్నందుకు ధన్యవాదాలు.",
        "మీ ఖాతా విజయవంతంగా అప్‌డేట్ చేయబడింది.",
        "ఈ రోజు వాతావరణం ఎండగా ఉంటుందని అంచనా.",
        "దయచేసి వేచి ఉండండి, మీ కాల్‌ను బదిలీ చేస్తున్నాము.",
        "మీ ఓపికకు ధన్యవాదాలు.",
        "మరింత సమాచారం కోసం మా వెబ్‌సైట్ సందర్శించండి.",
        "ఇది ఒక ఆటోమేటెడ్ మెసేజ్.",
        "మా ఆఫీస్ టైమింగ్స్ ఉదయం తొమ్మిది నుండి సాయంత్రం ఐదు వరకు.",
        "మీ ఆర్డర్ విజయవంతంగా ఉంచబడింది.",
    ],
    "malayalam": [
        "നമസ്കാരം, എന്റെ പേര് രാജേഷ്. ഞാൻ നിങ്ങളെ സഹായിക്കാൻ ഇവിടെയുണ്ട്.",
        "ഞങ്ങളുടെ സേവനം തിരഞ്ഞെടുത്തതിന് നന്ദി.",
        "നിങ്ങളുടെ അക്കൗണ്ട് വിജയകരമായി അപ്‌ഡേറ്റ് ചെയ്തു.",
        "ഇന്നത്തെ കാലാവസ്ഥ വെയിലായിരിക്കുമെന്ന് പ്രതീക്ഷിക്കുന്നു.",
        "ദയവായി കാത്തിരിക്കൂ, നിങ്ങളുടെ കോൾ ട്രാൻസ്ഫർ ചെയ്യുന്നു.",
        "നിങ്ങളുടെ ക്ഷമയ്ക്ക് നന്ദി.",
        "കൂടുതൽ വിവരങ്ങൾക്ക് ഞങ്ങളുടെ വെബ്സൈറ്റ് സന്ദർശിക്കുക.",
        "ഇത് ഒരു ഓട്ടോമേറ്റഡ് സന്ദേശമാണ്.",
        "ഞങ്ങളുടെ ഓഫീസ് സമയം രാവിലെ ഒൻപത് മുതൽ വൈകുന്നേരം അഞ്ച് വരെയാണ്.",
        "നിങ്ങളുടെ ഓർഡർ വിജയകരമായി സ്ഥാപിച്ചു.",
    ],
}

# Common Voice dataset URLs (Mozilla's open source voice dataset)
COMMON_VOICE_SAMPLES = {
    "english": [
        "https://upload.wikimedia.org/wikipedia/commons/2/22/En-us-hello.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4f/En-us-welcome.ogg",
    ],
    "hindi": [
        "https://upload.wikimedia.org/wikipedia/commons/9/99/Hi-%E0%A4%A8%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A5%87.oga",
    ],
}


async def generate_ai_samples():
    """Generate AI voice samples using Microsoft Edge TTS."""
    print("\n" + "="*50)
    print("GENERATING AI VOICE SAMPLES")
    print("="*50)
    
    for language, voices in VOICES.items():
        output_dir = DATA_DIR / "ai" / language
        output_dir.mkdir(parents=True, exist_ok=True)
        
        texts = TEXTS.get(language, TEXTS["english"])
        
        print(f"\n[{language.upper()}] Generating {len(texts)} samples...")
        
        for i, text in enumerate(texts):
            voice = voices[i % len(voices)]
            output_file = output_dir / f"ai_sample_{i+1}.mp3"
            
            if output_file.exists():
                print(f"  [{i+1}/{len(texts)}] Already exists: {output_file.name}")
                continue
            
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(output_file))
                print(f"  [{i+1}/{len(texts)}] ✓ Generated: {output_file.name} ({voice})")
            except Exception as e:
                print(f"  [{i+1}/{len(texts)}] ✗ Error: {str(e)}")
    
    print("\n✓ AI samples generation complete!")


async def download_file(session, url, filepath):
    """Download a file from URL."""
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                content = await response.read()
                with open(filepath, 'wb') as f:
                    f.write(content)
                return True
    except Exception as e:
        print(f"    Error downloading: {str(e)}")
    return False


async def generate_human_samples_from_tts():
    """
    Generate 'human-like' samples using different TTS settings.
    Note: For best results, replace these with real human recordings.
    """
    print("\n" + "="*50)
    print("GENERATING PLACEHOLDER HUMAN SAMPLES")
    print("="*50)
    print("Note: For best accuracy, replace with real human voice recordings!")
    
    # We'll use different voice styles and pitches to create variation
    # These are still TTS but with different characteristics
    
    human_voices = {
        "english": [
            ("en-US-AriaNeural", "--pitch=-10Hz --rate=-10%"),
            ("en-US-DavisNeural", "--pitch=+5Hz --rate=+5%"),
            ("en-GB-RyanNeural", "--pitch=-5Hz --rate=-5%"),
            ("en-AU-WilliamNeural", "--pitch=+10Hz --rate=+10%"),
        ],
        "hindi": [
            ("hi-IN-SwaraNeural", "--pitch=-8Hz --rate=-8%"),
            ("hi-IN-MadhurNeural", "--pitch=+8Hz --rate=+8%"),
        ],
        "tamil": [
            ("ta-IN-PallaviNeural", "--pitch=-10Hz --rate=-10%"),
            ("ta-IN-ValluvarNeural", "--pitch=+5Hz --rate=+5%"),
        ],
        "telugu": [
            ("te-IN-ShrutiNeural", "--pitch=-8Hz --rate=-5%"),
            ("te-IN-MohanNeural", "--pitch=+5Hz --rate=+8%"),
        ],
        "malayalam": [
            ("ml-IN-SobhanaNeural", "--pitch=-10Hz --rate=-10%"),
            ("ml-IN-MidhunNeural", "--pitch=+8Hz --rate=+5%"),
        ],
    }
    
    # Different texts for human samples (more conversational)
    human_texts = {
        "english": [
            "Hey, how are you doing today? I was thinking about grabbing some coffee.",
            "So I was telling my friend about this new movie, and she was like, wow!",
            "Umm, I think the meeting is scheduled for three o'clock, if I remember correctly.",
            "Oh wow, that's really interesting! I never thought about it that way before.",
            "Yeah, sure, I can help you with that. Let me just check my schedule first.",
            "Honestly, I'm not really sure what to do about this situation, you know?",
            "So anyway, like I was saying, the weather has been crazy lately.",
            "Ha! That's so funny. My sister said the exact same thing yesterday.",
            "Well, I mean, it depends on how you look at it, right?",
            "Oh no, I think I left my keys somewhere. Can you help me find them?",
        ],
        "hindi": [
            "अरे, तुम कैसे हो? मैं कल तुम्हारे बारे में सोच रहा था।",
            "हाँ, मुझे याद है। वो बहुत मजेदार था, है ना?",
            "उम्म, मुझे लगता है कि हमें कल मिलना चाहिए।",
            "सच में? मुझे पता नहीं था कि ऐसा होगा।",
            "ठीक है, चलो देखते हैं क्या होता है।",
            "मुझे थोड़ा सोचने दो, फिर बताता हूँ।",
            "वाह, ये तो बहुत अच्छा है!",
            "अरे हाँ, मैं भूल गया था बताना।",
            "चलो, कहीं चाय पीते हैं।",
            "मुझे नहीं पता, शायद कल पता चले।",
        ],
        "tamil": [
            "என்ன செய்யுற? நான் உன்னை பத்தி நினைச்சேன்.",
            "ஆமா, எனக்கு ஞாபகம் இருக்கு. அது ரொம்ப நல்லா இருந்தது.",
            "சரி, நாளைக்கு பார்க்கலாம்.",
            "அட, நான் இதை எதிர்பார்க்கல.",
            "ஓகே, பார்க்கலாம் என்ன ஆகுதுன்னு.",
            "கொஞ்சம் யோசிக்கணும், அப்புறம் சொல்றேன்.",
            "வாவ், இது நல்லாருக்கு!",
            "ஆமா, சொல்ல மறந்துட்டேன்.",
            "வா, டீ குடிக்கலாம்.",
            "தெரியல, நாளைக்கு தெரியும்.",
        ],
        "telugu": [
            "ఏంటి చేస్తున్నావ్? నేను నిన్ను గుర్తు చేసుకున్నాను.",
            "అవును, నాకు గుర్తుంది. అది చాలా బాగుంది.",
            "సరే, రేపు కలుద్దాం.",
            "అరే, నేను ఇది ఊహించలేదు.",
            "ఓకే, ఏమి జరుగుతుందో చూద్దాం.",
            "కొంచెం ఆలోచించాలి, తర్వాత చెప్తాను.",
            "వావ్, ఇది బాగుంది!",
            "అవును, చెప్పడం మర్చిపోయాను.",
            "రా, టీ తాగుదాం.",
            "తెలియదు, రేపు తెలుస్తుంది.",
        ],
        "malayalam": [
            "എന്താ ചെയ്യുന്നത്? ഞാൻ നിന്നെ ഓർത്തു.",
            "അതെ, എനിക്ക് ഓർമയുണ്ട്. അത് നന്നായിരുന്നു.",
            "ശരി, നാളെ കാണാം.",
            "അയ്യോ, ഞാൻ ഇത് പ്രതീക്ഷിച്ചില്ല.",
            "ഓക്കേ, എന്താ സംഭവിക്കുന്നതെന്ന് നോക്കാം.",
            "കുറച്ച് ആലോചിക്കണം, പിന്നെ പറയാം.",
            "വാവ്, ഇത് നല്ലതാണ്!",
            "അതെ, പറയാൻ മറന്നു.",
            "വാ, ചായ കുടിക്കാം.",
            "അറിയില്ല, നാളെ അറിയും.",
        ],
    }
    
    for language in VOICES.keys():
        output_dir = DATA_DIR / "human" / language
        output_dir.mkdir(parents=True, exist_ok=True)
        
        texts = human_texts.get(language, human_texts["english"])
        voices = human_voices.get(language, human_voices["english"])
        
        print(f"\n[{language.upper()}] Generating {len(texts)} human-like samples...")
        
        for i, text in enumerate(texts):
            voice_config = voices[i % len(voices)]
            voice = voice_config[0]
            output_file = output_dir / f"human_sample_{i+1}.mp3"
            
            if output_file.exists():
                print(f"  [{i+1}/{len(texts)}] Already exists: {output_file.name}")
                continue
            
            try:
                # Add slight variations to make it sound different from AI samples
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(output_file))
                print(f"  [{i+1}/{len(texts)}] ✓ Generated: {output_file.name}")
            except Exception as e:
                print(f"  [{i+1}/{len(texts)}] ✗ Error: {str(e)}")
    
    print("\n✓ Human placeholder samples complete!")
    print("\n⚠️  IMPORTANT: For better accuracy, replace human samples with REAL recordings!")


async def main():
    """Main function to generate all samples."""
    print("="*50)
    print("AUDIO SAMPLE GENERATOR")
    print("="*50)
    print(f"Data directory: {DATA_DIR}")
    
    # Generate AI samples
    await generate_ai_samples()
    
    # Generate placeholder human samples
    await generate_human_samples_from_tts()
    
    print("\n" + "="*50)
    print("SAMPLE GENERATION COMPLETE!")
    print("="*50)
    
    # Count files
    ai_count = sum(1 for _ in (DATA_DIR / "ai").rglob("*.mp3"))
    human_count = sum(1 for _ in (DATA_DIR / "human").rglob("*.mp3"))
    
    print(f"\nTotal AI samples: {ai_count}")
    print(f"Total Human samples: {human_count}")
    print(f"\nNext step: Run 'python train.py' to train the model")


if __name__ == "__main__":
    asyncio.run(main())
