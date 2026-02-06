"""
Malayalam Audio Dataset Generator
Generates 500 male and 500 female voice audio files with sensible Malayalam sentences
for machine learning purposes to identify AI tone.
"""

import asyncio
import edge_tts
import os
import random

# Create output directories
MALE_DIR = "male_audio"
FEMALE_DIR = "female_audio"

# Malayalam voices from Edge TTS
MALE_VOICE = "ml-IN-MidhunNeural"  # Malayalam Male voice
FEMALE_VOICE = "ml-IN-SobhanaNeural"  # Malayalam Female voice

# Sensible Malayalam sentences - diverse topics for ML training
MALAYALAM_SENTENCES = [
    # Greetings and daily life
    "നമസ്കാരം, സുഖമാണോ?",
    "ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്.",
    "നിങ്ങൾ എവിടെ പോകുന്നു?",
    "എനിക്ക് ഒരു ചായ വേണം.",
    "നാളെ ഞാൻ വരും.",
    "ദയവായി ഇരിക്കൂ.",
    "നന്ദി, വളരെ നന്നായി.",
    "എനിക്ക് മലയാളം സംസാരിക്കാൻ അറിയാം.",
    "ഈ പുസ്തകം വളരെ രസകരമാണ്.",
    "എന്റെ പേര് എന്താണെന്ന് അറിയാമോ?",
    
    # Food and cooking
    "കേരളത്തിലെ ഭക്ഷണം വളരെ രുചികരമാണ്.",
    "ഇന്നത്തെ ഉച്ചഭക്ഷണം എന്താണ്?",
    "എനിക്ക് ചോറും സാമ്പാറും ഇഷ്ടമാണ്.",
    "അമ്മ നല്ല പായസം ഉണ്ടാക്കി.",
    "പുട്ടും കടലയും കഴിക്കാം.",
    "ഈ കറി വളരെ എരിവുള്ളതാണ്.",
    "ചായയിൽ പഞ്ചസാര വേണോ?",
    "പഴങ്ങൾ ആരോഗ്യത്തിന് നല്ലതാണ്.",
    "വെള്ളം കുടിക്കുന്നത് നല്ലതാണ്.",
    "ഈ മീൻ കറി അസ്സലാണ്.",
    
    # Nature and environment
    "മഴ പെയ്യുന്നു, കുട എടുക്കൂ.",
    "കേരളം ദൈവത്തിന്റെ സ്വന്തം നാടാണ്.",
    "മലകളും കാടുകളും മനോഹരമാണ്.",
    "സൂര്യൻ ഉദിച്ചു കഴിഞ്ഞു.",
    "നക്ഷത്രങ്ങൾ തിളങ്ങുന്നു.",
    "പൂക്കൾ വളരെ സുന്ദരമാണ്.",
    "കടൽ തീരത്ത് പോകാം.",
    "മരങ്ങൾ നമുക്ക് ഓക്സിജൻ തരുന്നു.",
    "പക്ഷികൾ പാടുന്നത് കേൾക്കൂ.",
    "ഈ പൂന്തോട്ടം മനോഹരമാണ്.",
    
    # Family and relationships
    "എന്റെ കുടുംബം വളരെ വലുതാണ്.",
    "അച്ഛനും അമ്മയും എന്നെ സ്നേഹിക്കുന്നു.",
    "എന്റെ സഹോദരി പഠിക്കുന്നു.",
    "മുത്തശ്ശി കഥ പറയും.",
    "കുട്ടികൾ കളിക്കുകയാണ്.",
    "വീട്ടിൽ എല്ലാവരും സുഖമായിരിക്കുന്നു.",
    "സുഹൃത്തുക്കൾ വീട്ടിൽ വന്നു.",
    "നമ്മൾ ഒരുമിച്ച് പോകാം.",
    "എന്റെ മകൻ സ്കൂളിൽ പഠിക്കുന്നു.",
    "അമ്മയുടെ സ്നേഹം അതുല്യമാണ്.",
    
    # Education and learning
    "വിദ്യാഭ്യാസം വളരെ പ്രധാനമാണ്.",
    "ഞാൻ മലയാളം പഠിക്കുന്നു.",
    "പുസ്തകം വായിക്കുന്നത് നല്ലതാണ്.",
    "അധ്യാപകർ നമ്മെ പഠിപ്പിക്കുന്നു.",
    "പരീക്ഷ നാളെയാണ്.",
    "കമ്പ്യൂട്ടർ പഠിക്കുന്നത് ഉപകാരപ്രദമാണ്.",
    "ഗ്രന്ഥശാലയിൽ നിന്ന് പുസ്തകം എടുത്തു.",
    "എഴുതാനും വായിക്കാനും അറിയണം.",
    "ശാസ്ത്രം രസകരമായ വിഷയമാണ്.",
    "ചരിത്രം നമ്മെ പഠിപ്പിക്കുന്നു.",
    
    # Work and profession
    "ഞാൻ ഓഫീസിൽ പോകുന്നു.",
    "ജോലി ചെയ്യുന്നത് പ്രധാനമാണ്.",
    "ഡോക്ടർ രോഗികളെ ചികിത്സിക്കുന്നു.",
    "കർഷകർ കഠിനാധ്വാനികളാണ്.",
    "അധ്യാപകർ സമൂഹത്തിന്റെ തൂണുകളാണ്.",
    "എൻജിനീയർമാർ പാലം നിർമ്മിക്കുന്നു.",
    "പോലീസുകാർ നമ്മെ സംരക്ഷിക്കുന്നു.",
    "വ്യാപാരികൾ സാധനങ്ങൾ വിൽക്കുന്നു.",
    "മീൻപിടുത്തക്കാർ കടലിൽ പോകുന്നു.",
    "കലാകാരന്മാർ ചിത്രം വരയ്ക്കുന്നു.",
    
    # Travel and places
    "കേരളത്തിൽ സഞ്ചരിക്കാൻ നല്ലതാണ്.",
    "തിരുവനന്തപുരം തലസ്ഥാനമാണ്.",
    "കൊച്ചി വളരെ മനോഹരമായ നഗരമാണ്.",
    "മുന്നാർ ഒരു ഹിൽസ്റ്റേഷനാണ്.",
    "ആലപ്പുഴ കായൽ നാടാണ്.",
    "വയനാട്ടിലെ കാടുകൾ മനോഹരമാണ്.",
    "ബസ്സിൽ യാത്ര ചെയ്യാം.",
    "ട്രെയിനിൽ പോകുന്നത് സുഖകരമാണ്.",
    "വിമാനത്തിൽ യാത്ര ചെയ്യാം.",
    "നടന്നു പോകുന്നത് ആരോഗ്യത്തിന് നല്ലതാണ്.",
    
    # Health and wellness
    "ആരോഗ്യം വളരെ പ്രധാനമാണ്.",
    "വ്യായാമം ചെയ്യണം.",
    "നേരത്തെ ഉറങ്ങണം.",
    "പച്ചക്കറികൾ കഴിക്കണം.",
    "വൃത്തിയായി ഇരിക്കണം.",
    "കൈകൾ കഴുകണം.",
    "ഡോക്ടറെ കാണണം.",
    "മരുന്ന് കഴിക്കണം.",
    "വിശ്രമം ആവശ്യമാണ്.",
    "മാനസികാരോഗ്യം പ്രധാനമാണ്.",
    
    # Technology
    "കമ്പ്യൂട്ടർ ഉപയോഗിക്കുന്നു.",
    "ഇന്റർനെറ്റ് വളരെ ഉപകാരപ്രദമാണ്.",
    "മൊബൈൽ ഫോൺ എല്ലാവരുടെയും കൈയിലുണ്ട്.",
    "സാങ്കേതികവിദ്യ വളരുന്നു.",
    "ഓൺലൈനിൽ പഠിക്കാം.",
    "വീഡിയോ കോൾ ചെയ്യാം.",
    "ഇമെയിൽ അയയ്ക്കാം.",
    "സോഷ്യൽ മീഡിയ ഉപയോഗിക്കുന്നു.",
    "ആപ്പ് ഡൗൺലോഡ് ചെയ്യാം.",
    "ഡിജിറ്റൽ പേയ്മെന്റ് എളുപ്പമാണ്.",
    
    # Culture and traditions
    "ഓണം കേരളത്തിന്റെ ഉത്സവമാണ്.",
    "വിഷു പുതുവർഷമാണ്.",
    "കഥകളി ഒരു നൃത്തരൂപമാണ്.",
    "മോഹിനിയാട്ടം മനോഹരമാണ്.",
    "ക്ഷേത്രങ്ങൾ പവിത്രമാണ്.",
    "പള്ളികൾ പ്രാർത്ഥനാലയങ്ങളാണ്.",
    "മതസൗഹാർദ്ദം കേരളത്തിന്റെ സവിശേഷതയാണ്.",
    "സദ്യ ഓണത്തിന് ഉണ്ടാക്കും.",
    "തിരുവാതിര നൃത്തം മനോഹരമാണ്.",
    "പൂക്കളം ഓണത്തിന് ഉണ്ടാക്കും.",
    
    # Sports and games
    "കളിക്കുന്നത് ആരോഗ്യത്തിന് നല്ലതാണ്.",
    "ക്രിക്കറ്റ് ഇന്ത്യയിൽ പ്രശസ്തമാണ്.",
    "ഫുട്ബോൾ കേരളത്തിൽ പ്രിയപ്പെട്ടതാണ്.",
    "ഓട്ടം നല്ല വ്യായാമമാണ്.",
    "നീന്തൽ പഠിക്കണം.",
    "ചെസ്സ് ബുദ്ധിയുള്ള കളിയാണ്.",
    "കബഡി ഒരു ഇന്ത്യൻ കളിയാണ്.",
    "ബാഡ്മിന്റൺ കളിക്കാം.",
    "വോളിബോൾ ടീം കളിയാണ്.",
    "കുട്ടികൾ കളിക്കണം.",
    
    # Emotions and feelings
    "ഞാൻ സന്തോഷവാനാണ്.",
    "ഇന്ന് എനിക്ക് സങ്കടമുണ്ട്.",
    "നിങ്ങളെ കണ്ടതിൽ സന്തോഷം.",
    "അത് വളരെ രസകരമാണ്.",
    "എനിക്ക് ആശ്ചര്യമായി.",
    "നന്ദി പറയാൻ വാക്കുകളില്ല.",
    "ക്ഷമിക്കണം, തെറ്റ് പറ്റി.",
    "എനിക്ക് വിശ്വാസമുണ്ട്.",
    "പ്രതീക്ഷ കൈവിടരുത്.",
    "സ്നേഹം ലോകത്തെ മാറ്റും.",
    
    # Time and seasons
    "ഇപ്പോൾ രാവിലെയാണ്.",
    "ഉച്ചയ്ക്ക് ഊണ് കഴിക്കും.",
    "വൈകുന്നേരം നടക്കാൻ പോകും.",
    "രാത്രി നേരത്തെ ഉറങ്ങണം.",
    "മഴക്കാലം വരുന്നു.",
    "വേനൽക്കാലം ചൂടുള്ളതാണ്.",
    "ശൈത്യകാലം തണുപ്പുള്ളതാണ്.",
    "ഞായറാഴ്ച അവധിയാണ്.",
    "പുതുവർഷം വരുന്നു.",
    "സമയം വിലപ്പെട്ടതാണ്.",
    
    # Shopping and economy
    "കടയിൽ പോകണം.",
    "സാധനങ്ങൾ വാങ്ങണം.",
    "വില കൂടുതലാണ്.",
    "ഡിസ്കൗണ്ട് കിട്ടും.",
    "പണം സൂക്ഷിക്കണം.",
    "ബാങ്കിൽ പോകണം.",
    "ബിൽ അടയ്ക്കണം.",
    "സാമ്പത്തികം പ്രധാനമാണ്.",
    "ഓൺലൈൻ ഷോപ്പിംഗ് എളുപ്പമാണ്.",
    "കാർഡ് ഉപയോഗിക്കാം.",
    
    # Animals and pets
    "നായ വിശ്വസ്തനായ മൃഗമാണ്.",
    "പൂച്ച വീട്ടിലുണ്ട്.",
    "പക്ഷികൾ ആകാശത്ത് പറക്കുന്നു.",
    "ആന കേരളത്തിന്റെ അഭിമാനമാണ്.",
    "കന്നുകാലികൾ കർഷകന്റെ സ്വത്താണ്.",
    "മത്സ്യങ്ങൾ വെള്ളത്തിൽ നീന്തുന്നു.",
    "പശു പാൽ തരുന്നു.",
    "കുതിര വേഗത്തിൽ ഓടും.",
    "മയിൽ ദേശീയ പക്ഷിയാണ്.",
    "കാക്ക ബുദ്ധിയുള്ള പക്ഷിയാണ്.",
    
    # Home and daily activities
    "വീട് വൃത്തിയാക്കണം.",
    "അടുക്കളയിൽ പാചകം ചെയ്യുന്നു.",
    "കുളിക്കണം.",
    "തുണി അലക്കണം.",
    "പാത്രം കഴുകണം.",
    "മുറി അടുക്കി വെക്കണം.",
    "ടിവി കാണുന്നു.",
    "സംഗീതം കേൾക്കുന്നു.",
    "പൂന്തോട്ടത്തിൽ പണി ചെയ്യുന്നു.",
    "വിളക്ക് കത്തിക്കണം.",
    
    # Communication
    "സംസാരിക്കുന്നത് പ്രധാനമാണ്.",
    "ശ്രദ്ധിച്ച് കേൾക്കണം.",
    "മറുപടി പറയണം.",
    "ചോദ്യം ചോദിക്കാം.",
    "വിശദീകരിക്കാം.",
    "സംശയം തീർക്കാം.",
    "അഭിപ്രായം പറയാം.",
    "ചർച്ച ചെയ്യാം.",
    "സമ്മതിക്കുന്നു.",
    "എതിർക്കുന്നു.",
    
    # News and current affairs
    "വാർത്ത കേട്ടോ?",
    "പത്രം വായിക്കണം.",
    "ലോകത്ത് പലതും നടക്കുന്നു.",
    "രാഷ്ട്രീയം മാറുന്നു.",
    "സമൂഹം വികസിക്കുന്നു.",
    "പരിസ്ഥിതി സംരക്ഷിക്കണം.",
    "കാലാവസ്ഥ മാറ്റം ഗുരുതരമാണ്.",
    "സാങ്കേതികവിദ്യ മുന്നേറുന്നു.",
    "സമ്പദ്വ്യവസ്ഥ വളരുന്നു.",
    "വിദ്യാഭ്യാസം മെച്ചപ്പെടുന്നു.",
    
    # Wisdom and proverbs
    "അറിവ് സമ്പത്താണ്.",
    "സമയം പൊന്നാണ്.",
    "ക്ഷമ ഗുണമാണ്.",
    "ആരോഗ്യമാണ് യഥാർത്ഥ സമ്പത്ത്.",
    "ഐക്യത്തിൽ ശക്തിയുണ്ട്.",
    "പ്രവൃത്തി വാക്കിനേക്കാൾ വലുതാണ്.",
    "വിനയം വലിയ ഗുണമാണ്.",
    "സത്യം ജയിക്കും.",
    "അധ്വാനിക്കുന്നവൻ വിജയിക്കും.",
    "നന്മ ചെയ്യുക.",
    
    # Questions and conversations
    "നിങ്ങളുടെ പേരെന്താണ്?",
    "എവിടെയാണ് താമസം?",
    "എന്ത് ജോലിയാണ്?",
    "എങ്ങനെ പോകും?",
    "എപ്പോൾ വരും?",
    "എന്തിനാണ് പോകുന്നത്?",
    "ആർക്കാണ് വേണ്ടത്?",
    "ഏതാണ് നല്ലത്?",
    "എത്രയാണ് വില?",
    "എന്താണ് സംഭവിച്ചത്?",
    
    # Additional varied sentences
    "മലയാളം എന്റെ മാതൃഭാഷയാണ്.",
    "കേരളത്തിന്റെ സംസ്കാരം അതുല്യമാണ്.",
    "പ്രകൃതി സൗന്ദര്യം ആസ്വദിക്കണം.",
    "സുഹൃത്തുക്കളോടൊപ്പം സമയം ചെലവഴിക്കുന്നത് നല്ലതാണ്.",
    "ജീവിതം മനോഹരമാണ്.",
    "പുതിയ കാര്യങ്ങൾ പഠിക്കണം.",
    "സഹായിക്കാൻ തയ്യാറാണ്.",
    "ഒരുമിച്ച് പ്രവർത്തിക്കാം.",
    "എല്ലാവർക്കും ശുഭദിനം.",
    "നല്ല ഭാവി പ്രതീക്ഷിക്കുന്നു.",
    "മനസ്സ് ശാന്തമാക്കണം.",
    "യോഗ ചെയ്യുന്നത് നല്ലതാണ്.",
    "ധ്യാനം മനസ്സിന് ഗുണകരമാണ്.",
    "സംഗീതം ആത്മാവിന്റെ ഭാഷയാണ്.",
    "കല ജീവിതത്തിന്റെ ഭാഗമാണ്.",
    "സാഹിത്യം വായിക്കണം.",
    "കവിത എഴുതുന്നത് രസകരമാണ്.",
    "സിനിമ കാണാം.",
    "നാടകം അവതരിപ്പിക്കാം.",
    "ചിത്രം വരയ്ക്കാം.",
    "ഫോട്ടോ എടുക്കാം.",
    "യാത്ര ചെയ്യുന്നത് അനുഭവമാണ്.",
    "പുതിയ സ്ഥലങ്ങൾ കാണാം.",
    "ലോകം വലുതാണ്.",
    "ഭൂമി നമ്മുടെ വീടാണ്.",
    "ജലം ജീവനാണ്.",
    "വായു ശുദ്ധമായിരിക്കണം.",
    "മണ്ണ് വിലപ്പെട്ടതാണ്.",
    "ആകാശം നീലയാണ്.",
    "ചന്ദ്രൻ മനോഹരമാണ്.",
    "സൂര്യൻ ഊർജ്ജം തരുന്നു.",
    "വർഷം കഴിയുന്നു.",
    "പുതിയ തുടക്കം.",
    "ഓർമ്മകൾ മധുരമാണ്.",
    "സ്വപ്നങ്ങൾ സാക്ഷാത്കരിക്കാം.",
    "ലക്ഷ്യം നേടാം.",
    "വിജയിക്കാൻ പരിശ്രമിക്കണം.",
    "പരാജയം വിജയത്തിന്റെ ആദ്യപടിയാണ്.",
    "ഉത്സാഹത്തോടെ പ്രവർത്തിക്കണം.",
    "സന്തോഷത്തോടെ ജീവിക്കണം.",
    "സമാധാനം പ്രധാനമാണ്.",
    "സ്നേഹം പങ്കുവെക്കണം.",
    "കരുണ കാണിക്കണം.",
    "ദയ ഉണ്ടായിരിക്കണം.",
    "ബഹുമാനം നൽകണം.",
    "വിശ്വാസം കാത്തുസൂക്ഷിക്കണം.",
    "സത്യസന്ധത പ്രധാനമാണ്.",
    "നീതി നടപ്പാക്കണം.",
    "സമത്വം ഉറപ്പാക്കണം.",
    "സ്വാതന്ത്ര്യം വിലമതിക്കണം.",
    "ഉത്തരവാദിത്വം ഏറ്റെടുക്കണം.",
]

# Additional sentence templates for variety
SENTENCE_TEMPLATES = [
    "ഇന്ന് {} ചെയ്യണം.",
    "നാളെ {} ഉണ്ട്.",
    "{} വളരെ പ്രധാനമാണ്.",
    "എനിക്ക് {} ഇഷ്ടമാണ്.",
    "{} കഴിഞ്ഞു.",
    "ദയവായി {} ചെയ്യൂ.",
    "{} എവിടെയാണ്?",
    "{} എങ്ങനെയാണ്?",
    "നിങ്ങൾ {} അറിയുമോ?",
    "{} നല്ലതാണ്.",
]

TEMPLATE_FILLERS = [
    "പഠനം", "ജോലി", "യാത്ര", "ഭക്ഷണം", "വിശ്രമം",
    "വ്യായാമം", "വായന", "എഴുത്ത്", "സംഗീതം", "കല",
    "കൃഷി", "പാചകം", "നടത്തം", "നീന്തൽ", "ഓട്ടം",
    "ചിന്ത", "ധ്യാനം", "പ്രാർത്ഥന", "സേവനം", "സഹായം",
]

def generate_varied_sentences(count):
    """Generate varied sentences by combining templates and fillers with base sentences."""
    sentences = list(MALAYALAM_SENTENCES)
    
    # Add template-based sentences
    for _ in range(count - len(MALAYALAM_SENTENCES)):
        template = random.choice(SENTENCE_TEMPLATES)
        filler = random.choice(TEMPLATE_FILLERS)
        sentences.append(template.format(filler))
    
    # Shuffle and return
    random.shuffle(sentences)
    return sentences[:count]

async def generate_audio(text, voice, output_file, rate="+0%"):
    """Generate audio file using edge-tts."""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_file)

async def generate_dataset():
    """Generate the complete audio dataset."""
    # Create directories
    os.makedirs(MALE_DIR, exist_ok=True)
    os.makedirs(FEMALE_DIR, exist_ok=True)
    
    print("="*60)
    print("Malayalam Audio Dataset Generator")
    print("="*60)
    print(f"Generating 500 male voice audio files in '{MALE_DIR}/'")
    print(f"Generating 500 female voice audio files in '{FEMALE_DIR}/'")
    print("="*60)
    
    # Generate sentences
    male_sentences = generate_varied_sentences(500)
    female_sentences = generate_varied_sentences(500)
    
    # Generate male audio files
    print("\n[1/2] Generating male voice audio files...")
    for i, sentence in enumerate(male_sentences, 1):
        output_file = os.path.join(MALE_DIR, f"male_{i:04d}.mp3")
        try:
            await generate_audio(sentence, MALE_VOICE, output_file)
            if i % 50 == 0 or i == 1:
                print(f"  Progress: {i}/500 files generated")
        except Exception as e:
            print(f"  Error generating {output_file}: {e}")
    
    print(f"  ✓ Completed: 500 male audio files")
    
    # Generate female audio files
    print("\n[2/2] Generating female voice audio files...")
    for i, sentence in enumerate(female_sentences, 1):
        output_file = os.path.join(FEMALE_DIR, f"female_{i:04d}.mp3")
        try:
            await generate_audio(sentence, FEMALE_VOICE, output_file)
            if i % 50 == 0 or i == 1:
                print(f"  Progress: {i}/500 files generated")
        except Exception as e:
            print(f"  Error generating {output_file}: {e}")
    
    print(f"  ✓ Completed: 500 female audio files")
    
    # Create metadata file
    print("\n[Creating metadata...]")
    with open("dataset_metadata.txt", "w", encoding="utf-8") as f:
        f.write("Malayalam Audio Dataset Metadata\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Files: 1000\n")
        f.write(f"Male Voice Files: 500 (in '{MALE_DIR}/' folder)\n")
        f.write(f"Female Voice Files: 500 (in '{FEMALE_DIR}/' folder)\n")
        f.write(f"Voice: Edge TTS Neural Voices\n")
        f.write(f"Male Voice ID: {MALE_VOICE}\n")
        f.write(f"Female Voice ID: {FEMALE_VOICE}\n")
        f.write(f"Language: Malayalam (ml-IN)\n")
        f.write(f"Format: MP3\n")
        f.write(f"Speed: Normal/Average\n\n")
        f.write("="*50 + "\n")
        f.write("MALE AUDIO FILES AND SENTENCES\n")
        f.write("="*50 + "\n\n")
        for i, sentence in enumerate(male_sentences, 1):
            f.write(f"male_{i:04d}.mp3: {sentence}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("FEMALE AUDIO FILES AND SENTENCES\n")
        f.write("="*50 + "\n\n")
        for i, sentence in enumerate(female_sentences, 1):
            f.write(f"female_{i:04d}.mp3: {sentence}\n")
    
    print("\n" + "="*60)
    print("✓ Dataset generation complete!")
    print("="*60)
    print(f"\nOutput:")
    print(f"  • Male audio files: {MALE_DIR}/male_0001.mp3 to male_0500.mp3")
    print(f"  • Female audio files: {FEMALE_DIR}/female_0001.mp3 to female_0500.mp3")
    print(f"  • Metadata: dataset_metadata.txt")
    print("\nTotal: 1000 audio files generated for ML training")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(generate_dataset())
