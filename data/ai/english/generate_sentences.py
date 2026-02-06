import pyttsx3
import random
import os

# Pre-written coherent sentences that make sense and are 5+ seconds at normal speed
# Each sentence is meaningful, grammatically correct, and tells a complete thought
sensible_sentences = [
    # Daily life and routines
    "Every morning, I wake up early and make myself a fresh cup of coffee before starting my day.",
    "The children walked to school together, laughing and talking about their weekend adventures.",
    "She carefully prepared dinner for her family while listening to her favorite music on the radio.",
    "After a long day at work, he enjoyed relaxing on the couch with a good book in his hands.",
    "The neighbors gathered in the park every Sunday afternoon for a friendly game of soccer.",
    "My grandmother always told me that hard work and patience are the keys to success in life.",
    "The students studied hard for their final exams, hoping to get good grades this semester.",
    "We decided to go for a long walk in the countryside to enjoy the beautiful autumn weather.",
    "The doctor advised her to drink more water and get plenty of rest to recover quickly.",
    "Every evening, the family sits together at the dinner table to share stories about their day.",
    
    # Nature and weather
    "The sun was setting behind the mountains, painting the sky in beautiful shades of orange and pink.",
    "A gentle breeze blew through the trees, carrying the sweet scent of flowers across the garden.",
    "The rain started falling softly at first, then grew heavier as dark clouds covered the sky.",
    "Birds sang cheerfully in the early morning, welcoming the arrival of another beautiful spring day.",
    "The river flowed peacefully through the valley, reflecting the bright blue sky above like a mirror.",
    "Snowflakes danced gracefully in the cold winter air before settling on the frozen ground below.",
    "The forest was quiet and peaceful, with only the sound of leaves rustling in the wind.",
    "After the storm passed, a beautiful rainbow appeared across the sky, bringing smiles to everyone.",
    "The waves crashed gently against the shore as the sun slowly rose over the calm ocean.",
    "In autumn, the leaves change from green to brilliant shades of red, orange, and golden yellow.",
    
    # Travel and adventure
    "We packed our bags and set off on an exciting journey to explore new places and meet new people.",
    "The airplane soared high above the clouds, carrying passengers to destinations around the world.",
    "After driving for several hours, we finally arrived at the beautiful mountain resort for our vacation.",
    "The tour guide explained the fascinating history of the ancient castle to the curious visitors.",
    "She always dreamed of traveling to distant countries and experiencing different cultures firsthand.",
    "The hikers climbed steadily up the steep mountain trail, enjoying breathtaking views along the way.",
    "We took a boat ride across the lake, admiring the stunning scenery from the calm water.",
    "The train traveled through beautiful countryside, passing farms, villages, and rolling green hills.",
    "After exploring the busy city streets all day, we found a quiet cafe to rest our tired feet.",
    "The adventurous group decided to camp under the stars and tell stories around the warm fire.",
    
    # Work and education
    "The teacher explained the difficult concept several times until all the students understood it clearly.",
    "He worked overtime every day this week to finish the important project before the deadline.",
    "The company is looking for talented individuals who are passionate about technology and innovation.",
    "She attended a workshop to learn new skills that would help her advance in her career.",
    "The meeting lasted for two hours as the team discussed plans for the upcoming product launch.",
    "University students often spend long hours in the library researching and writing their papers.",
    "The manager praised the team for their excellent work and dedication to the company's success.",
    "After years of hard work and dedication, she finally achieved her dream of becoming a doctor.",
    "The new employee quickly learned the company procedures and became a valuable member of the team.",
    "Online courses have made it easier for people to learn new subjects from the comfort of home.",
    
    # Food and cooking
    "The chef prepared a delicious meal using fresh ingredients from the local farmers market today.",
    "She followed her mother's recipe carefully to make the perfect chocolate cake for the birthday party.",
    "The restaurant was famous for its homemade pasta and freshly baked bread that melted in your mouth.",
    "We gathered in the kitchen to cook a special dinner together and celebrate the holiday season.",
    "The aroma of freshly brewed coffee filled the room, making everyone feel warm and comfortable.",
    "He learned to cook from watching online videos and reading cookbooks during the quarantine period.",
    "The bakery down the street makes the most delicious pastries and cakes in the entire neighborhood.",
    "She always brings homemade cookies to share with her colleagues at the office every Friday afternoon.",
    "The farmers market offers a wonderful variety of fresh fruits, vegetables, and organic products weekly.",
    "Nothing tastes better than a home-cooked meal shared with family and friends on a special occasion.",
    
    # Health and wellness
    "Regular exercise and a balanced diet are essential for maintaining good health throughout your life.",
    "The doctor recommended taking a short walk every day to improve circulation and reduce stress levels.",
    "Getting enough sleep is important for your body to recover and prepare for the next day ahead.",
    "She started practicing yoga and meditation to find inner peace and improve her mental wellbeing.",
    "Drinking plenty of water throughout the day helps keep your body hydrated and functioning properly.",
    "The hospital staff worked tirelessly day and night to care for patients during the busy season.",
    "Mental health is just as important as physical health, so please take care of your mind too.",
    "He decided to quit smoking and start living a healthier lifestyle for his family's sake.",
    "The gym offers various fitness classes, including swimming, dancing, and strength training programs.",
    "Taking breaks during work hours can help reduce eye strain and improve your overall productivity.",
    
    # Technology and modern life
    "Smartphones have changed the way we communicate, making it easier to stay connected with loved ones.",
    "The internet provides access to endless information and educational resources from around the world.",
    "She spent the afternoon learning how to use the new software program for her upcoming project.",
    "Social media allows people to share their experiences and connect with friends across great distances.",
    "The latest technology has made our daily lives more convenient, but also more complicated sometimes.",
    "Electric cars are becoming more popular as people look for environmentally friendly transportation options.",
    "He set up the new computer system and helped everyone learn how to use it efficiently at work.",
    "Online shopping has become increasingly popular because it saves time and offers great convenience.",
    "The company developed an innovative app that helps people manage their finances and save money wisely.",
    "Video calls have made it possible to have face-to-face meetings with people from different countries.",
    
    # Emotions and relationships
    "True friendship is built on trust, honesty, and mutual respect between people who care about each other.",
    "She felt grateful for all the support she received from her family during the challenging times.",
    "The couple celebrated their wedding anniversary with a romantic dinner at their favorite restaurant.",
    "Kindness and compassion can make a big difference in someone's life, so always be nice to others.",
    "He apologized sincerely for his mistake and promised to be more careful in the future from now on.",
    "The children were excited to see their grandparents after not visiting them for several months.",
    "She wrote a heartfelt letter to thank her teacher for all the guidance and inspiration over the years.",
    "Family gatherings are special occasions where we create lasting memories and strengthen our bonds together.",
    "He learned that listening carefully to others is just as important as expressing your own opinions.",
    "The community came together to help those in need after the unexpected disaster struck the town.",
    
    # Hobbies and interests
    "Playing musical instruments is a wonderful way to express yourself and develop creative skills over time.",
    "She spends her weekends painting beautiful landscapes and portraits in her small home studio space.",
    "Reading books opens up new worlds and helps expand your imagination beyond your everyday experiences.",
    "The photography club meets every month to share tips and showcase their best pictures from recent trips.",
    "He enjoys gardening in his backyard, growing vegetables and flowers that brighten up the whole neighborhood.",
    "The local theater group is preparing for their annual performance, rehearsing every evening after work.",
    "Learning a new language takes time and practice, but it opens doors to new cultures and opportunities.",
    "She joined a book club to discuss interesting stories and meet other people who love reading novels.",
    "The art museum displays beautiful works from artists around the world, attracting many visitors each year.",
    "Collecting stamps from different countries has been his favorite hobby since he was a young child.",
    
    # Dreams and aspirations
    "She dreams of opening her own business one day and becoming a successful entrepreneur in her field.",
    "Hard work and determination are necessary to achieve your goals and make your dreams come true someday.",
    "He set ambitious goals for himself and worked steadily toward achieving them throughout the entire year.",
    "The young athlete trained every day, hoping to compete in the international championships next summer.",
    "She believes that everyone has the potential to make a positive difference in the world around them.",
    "With patience and persistence, you can overcome any obstacle that stands between you and your dreams.",
    "The students shared their hopes and dreams for the future during the graduation ceremony last week.",
    "He never gave up on his dream of becoming a pilot, even when faced with many challenges along the way.",
    "Setting small achievable goals can help you stay motivated on the path toward bigger accomplishments.",
    "She inspired others with her story of overcoming difficulties and achieving success through hard work.",
    
    # Science and discovery
    "Scientists are working hard to find solutions to some of the world's most pressing problems today.",
    "The discovery of new planets in distant galaxies has sparked excitement among astronomers worldwide.",
    "Research and experimentation are essential parts of the scientific method for understanding our world.",
    "The museum exhibit explained how ancient civilizations developed technology and built amazing structures.",
    "Climate change is a serious issue that requires immediate action from governments and individuals alike.",
    "The laboratory team conducted experiments to test their hypothesis about the chemical reaction process.",
    "Space exploration has led to many technological advances that benefit our daily lives in unexpected ways.",
    "Scientists study the natural world to understand how things work and improve our quality of life.",
    "The documentary explained complex scientific concepts in a way that everyone could easily understand.",
    "New medical discoveries are helping doctors treat diseases that were once considered incurable conditions.",
    
    # History and culture
    "Learning about history helps us understand how past events have shaped the world we live in today.",
    "The ancient ruins tell stories of civilizations that flourished thousands of years before our time.",
    "Cultural traditions are passed down from generation to generation, keeping our heritage alive and strong.",
    "The history museum houses artifacts that give us a glimpse into how people lived in earlier times.",
    "She studied different cultures and their customs to gain a better understanding of the diverse world.",
    "Historical events have taught us valuable lessons about the importance of peace and cooperation globally.",
    "The festival celebrates the rich cultural heritage of the region with music, dance, and traditional food.",
    "Preserving historical buildings and monuments helps future generations learn about their ancestors' lives.",
    "The professor gave a fascinating lecture about the rise and fall of ancient empires throughout history.",
    "Understanding different perspectives on historical events helps us become more thoughtful and informed citizens.",
    
    # Animals and pets
    "The loyal dog waited patiently by the door every day for his owner to return home from work.",
    "Cats are independent creatures, but they also enjoy spending time with their human companions at home.",
    "The wildlife sanctuary provides a safe haven for injured animals to recover and return to nature.",
    "She volunteers at the animal shelter every weekend, helping to care for dogs and cats in need.",
    "Birds migrate thousands of miles every year, following the same routes their ancestors have used for ages.",
    "The veterinarian examined the sick puppy carefully and prescribed medicine to help it feel better soon.",
    "Marine life in the ocean is incredibly diverse, with thousands of species yet to be discovered underwater.",
    "Elephants are known for their excellent memory and strong social bonds within their family groups.",
    "The zoo educates visitors about wildlife conservation and the importance of protecting endangered species.",
    "Taking care of a pet requires responsibility, patience, and a lot of love and attention every day.",
    
    # Sports and fitness
    "The team practiced together for months, developing strategies and building their skills for the tournament.",
    "Running is a great way to stay fit and clear your mind after a stressful day at work.",
    "The championship game was exciting, with both teams playing their best until the final whistle blew.",
    "Swimming is an excellent full-body exercise that is gentle on the joints and good for cardiovascular health.",
    "She set a personal record in the marathon, crossing the finish line with a huge smile on her face.",
    "The coach encouraged the players to work as a team and support each other throughout the entire season.",
    "Cycling through the countryside is a wonderful way to explore nature while getting good exercise outdoors.",
    "The sports center offers programs for all ages, from children's classes to adult fitness training sessions.",
    "Watching the game with friends and family is a popular weekend activity that brings people together.",
    "Regular physical activity not only improves your health but also boosts your mood and energy levels.",
    
    # Home and family
    "The family spent the weekend redecorating the living room, choosing new colors and furniture together.",
    "Home is where we feel safe and comfortable, surrounded by the people and things we love the most.",
    "She organized the closets and donated clothes that no longer fit to a local charity organization.",
    "The garden behind the house is filled with beautiful flowers that bloom in different seasons throughout the year.",
    "Parents try their best to provide a loving and supportive environment for their children to grow up in.",
    "Weekend mornings are perfect for making a big breakfast and enjoying it slowly with the whole family.",
    "The cozy fireplace in the living room makes cold winter evenings feel warm and comfortable for everyone.",
    "They renovated the old kitchen to create a modern space where the family could cook and eat together.",
    "Family traditions, like holiday dinners and birthday celebrations, create special memories that last forever.",
    "Moving to a new house is exciting but also challenging, as it takes time to settle in and feel at home.",
    
    # Community and society
    "Volunteering in your community is a meaningful way to give back and help those who need assistance.",
    "The neighborhood organized a cleanup day to keep the streets and parks beautiful for everyone to enjoy.",
    "Public libraries provide free access to books, computers, and educational programs for all community members.",
    "The local government invested in new parks and recreational facilities to improve the quality of life.",
    "Community events bring people together and help neighbors get to know each other better over time.",
    "Small businesses are the backbone of local economies, providing jobs and services to their communities.",
    "The charity organization raises funds to support families in need and provide essential services to them.",
    "Neighbors looked out for each other during difficult times, offering help and support when it was needed.",
    "The town hall meeting gave residents a chance to voice their opinions and concerns about local issues.",
    "Building a strong community requires cooperation, mutual respect, and a willingness to help one another.",
    
    # Seasons and holidays
    "The holiday season brings families together to celebrate, exchange gifts, and enjoy special meals together.",
    "Spring is a time of renewal when flowers bloom and trees start growing fresh green leaves again.",
    "Summer vacations are perfect for spending time at the beach, swimming in the ocean, and relaxing outdoors.",
    "The autumn harvest festival celebrates the hard work of farmers who grow the food we eat every day.",
    "Winter brings cold weather and snow, making it the perfect season for skiing and building snowmen outside.",
    "New Year's Eve is a time to reflect on the past year and make resolutions for the year ahead.",
    "The colorful decorations and festive music create a joyful atmosphere during the holiday celebration season.",
    "Each season has its own unique beauty and activities that people look forward to throughout the year.",
    "The parade marched through the city streets, celebrating the national holiday with music, floats, and dancers.",
    "Spending time with loved ones during the holidays creates precious memories that we cherish for a lifetime.",
    
    # Arts and entertainment
    "The concert was amazing, with the orchestra playing beautiful music that moved the entire audience deeply.",
    "Going to the movies is a popular way to relax and enjoy a good story on the big screen.",
    "The art exhibition featured stunning paintings and sculptures from talented artists around the world today.",
    "She enjoys listening to different types of music, from classical symphonies to modern pop songs regularly.",
    "The theater performance told a moving story that made the audience laugh, cry, and think about life.",
    "Television shows and movies can educate us, entertain us, and help us understand different perspectives too.",
    "The dance performance was breathtaking, with the dancers moving gracefully across the stage in perfect harmony.",
    "Photography captures special moments in time, preserving memories that we can look back on for years.",
    "The museum's collection includes masterpieces from famous artists spanning several centuries of art history.",
    "Creativity is a powerful force that allows people to express themselves and share their vision with others.",
    
    # Learning and growth
    "Making mistakes is a natural part of learning, so don't be afraid to try new things and fail sometimes.",
    "Reading widely expands your knowledge and helps you see the world from many different points of view.",
    "The best teachers inspire their students to be curious and never stop asking questions about everything.",
    "Learning from experienced mentors can help you avoid common mistakes and accelerate your personal growth.",
    "Education opens doors to new opportunities and helps people build better futures for themselves and families.",
    "She took an online course to improve her skills and stay competitive in her rapidly changing industry.",
    "Critical thinking and problem-solving are essential skills that can be developed through practice and study.",
    "The workshop taught participants practical strategies for managing their time more effectively at work and home.",
    "Continuous learning keeps your mind sharp and helps you adapt to the constantly changing world around us.",
    "Sharing knowledge with others not only helps them grow but also reinforces your own understanding of subjects.",
]

def generate_sentence():
    return random.choice(sensible_sentences)

engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Find male and female voices
male_voice = None
female_voice = None
for voice in voices:
    if 'david' in voice.name.lower() or 'male' in voice.name.lower():
        male_voice = voice
    if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
        female_voice = voice

if not male_voice:
    male_voice = voices[0]  # Default
if not female_voice:
    female_voice = voices[1] if len(voices) > 1 else voices[0]

# Set average natural speaking rate (around 150 wpm - not too fast, not too slow)
engine.setProperty('rate', 145)  # Natural conversational speed
engine.setProperty('volume', 1.0)  # Full volume for clarity

# Generate male voices
engine.setProperty('voice', male_voice.id)
for i in range(1, 501):
    sentence = generate_sentence()
    filename = f"male_{i}.wav"
    engine.save_to_file(sentence, filename)
    print(f"Generated male_{i}.wav")

engine.runAndWait()

# Generate female voices
engine.setProperty('voice', female_voice.id)
for i in range(1, 501):
    sentence = generate_sentence()
    filename = f"female_{i}.wav"
    engine.save_to_file(sentence, filename)
    print(f"Generated female_{i}.wav")

engine.runAndWait()

print("All WAV files generated.")