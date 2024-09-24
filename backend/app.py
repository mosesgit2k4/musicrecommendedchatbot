from flask import Flask, request, jsonify
from flask_cors import CORS  # type: ignore
import pandas as pd
import random
import re
from rake_nltk import Rake
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import neattext.functions as nfx

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load datasets
songs_df = pd.read_csv('Content_Based_Filtering_T.csv')
emotion_df = pd.read_csv('emotion_dataset.csv')

# Preprocess emotion dataset
emotion_df['Clean_text'] = emotion_df['Text'].apply(nfx.remove_stopwords)
emotion_df['Clean_text'] = emotion_df['Clean_text'].apply(nfx.remove_punctuations)
emotion_df['Clean_text'] = emotion_df['Clean_text'].apply(nfx.remove_userhandles)

# Map emotions for simplicity
emotion_df['Emotion'] = emotion_df['Emotion'].replace({
    'shame': 'sadness',
    'fear': 'sadness',
    'disgust': 'sadness',
    'surprise': 'joy'
})

# Train emotion prediction model
Xfeatures = emotion_df['Clean_text']
ylabel = emotion_df['Emotion']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
X_train, X_test, y_train, y_test = train_test_split(X, ylabel, test_size=0.3, random_state=42)
nv_model = MultinomialNB()
nv_model.fit(X_train, y_train)

# Prepare song lists based on moods
funsong = songs_df[songs_df['Mood'] == 'Fun']['Song Title'].tolist()
enersong = songs_df[songs_df['Mood'] == 'Energetic']['Song Title'].tolist()

# Chatbot Class
class InteractiveConversationalBot:
    def __init__(self):
        self.memory = defaultdict(list)  # To remember conversation context
        self.user_name = None
        self.default_responses = [
            "Could you tell me more about that?",
            "That sounds interesting! What else?",
            "And then what happened?",
            "Do you often think about this?",
            "How does that make you feel?"
        ]
        self.engagement_questions = [
            "What do you think about this?",
            "How does this make you feel?",
            "Can you share more details?",
            "What are your thoughts on this?"
        ]
        self.greeting_responses = ["What's your name?", "How can I help you today?"]

    def extract_keywords(self, text):
        # Basic keyword extraction: remove stopwords and simple text processing
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lower the case
        rake = Rake()  # Initialize RAKE
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()
        return keywords

    def recommend_songs(self, keywords):
        # Recommend songs based on the extracted keywords
        recommended_songs = []
        for keyword in keywords:
            sampled_songs = songs_df.sample(5)
            for _, row in sampled_songs.iterrows():
                song_title = row['Song Title']
                artist_name = row['Artists']
                recommended_songs.append(f"{song_title} by {artist_name}")
        return recommended_songs

    def get_response(self, user_input):
        # Handle specific commands for ending the chat
        if user_input.lower() in ["bye", "quit", "exit"]:
            return "Thank you for chatting! Goodbye!"

        # Handle greetings
        if user_input.lower() in ["hi", "hello"]:
            return random.choice(self.greeting_responses)
        
        # If the user's name hasn't been set, ask for it
        if not self.user_name:
            self.user_name = user_input
            return f"Nice to meet you, {self.user_name}! What would you like to talk about today?"
        
        # Extract keywords and recommend songs
        keywords = self.extract_keywords(user_input)
        if keywords:
            recommended_songs = self.recommend_songs(keywords)
            response = f"I see you're interested in: {', '.join(keywords)}.\nHere are some song recommendations for you:\n"
            response += '\n'.join(recommended_songs)
            return response
        
        # If no keywords are found, reply with a random engagement response
        return random.choice(self.default_responses + self.engagement_questions)

# Create an instance of the chatbot
bot = InteractiveConversationalBot()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    # Generate the chatbot's response
    chatbot_response = bot.get_response(user_input)
    
    # Predict the emotion of the user input
    emotion_prediction = predict_emotion(user_input, nv_model)
    
    # Recommend a song based on the predicted emotion
    song_recommendation = display_song(emotion_prediction)
    
    # Send the chatbot's response and song recommendation back to the frontend
    return jsonify({
        "chatbot_response": chatbot_response,
        "recommended_songs": [song_recommendation]
    })

def predict_emotion(text, model):
    vect = cv.transform([text]).toarray()
    prediction = model.predict(vect)
    pred_proba = model.predict_proba(vect)
    pred_percentage_for_all = dict(zip(model.classes_, pred_proba[0]))
    return max(pred_percentage_for_all, key=pred_percentage_for_all.get)

def display_song(emotion):
    if emotion == 'joy':
        song = random.choice(funsong)
    elif emotion == 'sadness':
        song = random.choice(enersong)
    elif emotion == 'anger':
        song = random.choice(funsong)
    else:
        song = random.choice(funsong + enersong)
    
    # Get the artist name for the selected song
    artist_name = songs_df[songs_df['Song Title'] == song]['Artists'].values[0]
    return f"{song} by {artist_name}"

if __name__ == '__main__':
    port = 5000
    print(f"Starting server... Server is running on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
