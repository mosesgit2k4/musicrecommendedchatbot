import random
import re
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import pandas as pd
from rake_nltk import Rake
from textblob import TextBlob
import spacy # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import neattext.functions as nfx

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load datasets
songs_df = pd.read_csv('Content_Based_Filtering_T.csv')
emotion_df = pd.read_csv('emotion_dataset.csv')

# Preprocess emotion dataset
emotion_df['Clean_text'] = emotion_df['Text'].apply(nfx.remove_stopwords)
emotion_df['Clean_text'] = emotion_df['Clean_text'].apply(nfx.remove_punctuations)
emotion_df['Clean_text'] = emotion_df['Clean_text'].apply(nfx.remove_userhandles)

# Simplify emotion categories
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

    def identify_names(self, text):
        doc = nlp(text)
        people_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        organization_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        location_names = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

        return {
            "People Names": list(set(people_names)),
            "Organization Names": list(set(organization_names)),
            "Location Names": list(set(location_names))
        }

    def extract_keywords(self, text):
        rake = Rake()
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()
        return keywords if keywords else []

    def generate_followup_question(self, keyword):
        questions = [
            f"What do you think about {keyword}?",
            f"How do you feel about {keyword}?",
            f"Can you elaborate more on {keyword}?",
            f"Do you often deal with {keyword}?"
        ]
        return random.choice(questions)

    def generate_entity_based_question(self, entities):
        questions = []
        if entities["People Names"]:
            person = random.choice(entities["People Names"])
            questions.append(f"How do you know {person}?")
            questions.append(f"Is {person} someone important to you?")
        if entities["Organization Names"]:
            org = random.choice(entities["Organization Names"])
            questions.append(f"Have you worked with {org} before?")
            questions.append(f"What do you think about {org}?")
        if entities["Location Names"]:
            loc = random.choice(entities["Location Names"])
            questions.append(f"Have you ever visited {loc}?")
            questions.append(f"What do you like most about {loc}?")
        return random.choice(questions) if questions else None

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def personalize_response(self, text):
        sentiment = self.analyze_sentiment(text)
        if sentiment > 0.5:
            return "I'm glad to hear that!"
        elif sentiment < -0.5:
            return "I'm sorry to hear that. Do you want to talk more about it?"
        else:
            return random.choice(self.engagement_questions)

    def get_response(self, user_input):
        if user_input.lower() in ["bye", "quit", "exit"]:
            return "Thank you for chatting! Goodbye!"

        # Handle greetings and ask for the user's name
        if user_input.lower() in ["hi", "hello"]:
            return "What's your name?"
        if not self.user_name:
            self.user_name = user_input  # Capture the user's name
            return f"Nice to meet you, {self.user_name}! What would you like to talk about today?"

        entities = self.identify_names(user_input)
        entity_question = self.generate_entity_based_question(entities)
        if entity_question:
            return entity_question

        keywords = self.extract_keywords(user_input)
        if keywords:
            keyword = random.choice(keywords)
            if keyword not in self.memory:
                self.memory[keyword].append(user_input)
                followup = self.generate_followup_question(keyword)
                return f"{self.personalize_response(user_input)} {followup}"

        return random.choice(self.default_responses)


# Create an instance of the chatbot
bot = InteractiveConversationalBot()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    # Generate the chatbot's response
    chatbot_response = bot.get_response(user_input)

    # Predict the emotion of the user input
    emotion_prediction = predict_emotion(user_input, nv_model)

    # Recommend songs based on the predicted emotion
    song_recommendations = recommend_songs(emotion_prediction)

    # Send the chatbot's response and song recommendations back to the frontend
    return jsonify({
        "chatbot_response": chatbot_response,
        "recommended_songs": song_recommendations  # Return 3-5 recommended songs
    })


def predict_emotion(text, model):
    vect = cv.transform([text]).toarray()
    prediction = model.predict(vect)
    return prediction[0]


def recommend_songs(emotion):
    if emotion == 'joy':
        songs = random.sample(funsong, k=min(5, len(funsong)))
    elif emotion == 'sadness':
        songs = random.sample(enersong, k=min(5, len(enersong)))
    else:
        songs = random.sample(funsong + enersong, k=min(5, len(funsong + enersong)))
    
    recommended_songs = []
    for song in songs:
        artist_name = songs_df[songs_df['Song Title'] == song]['Artists'].values[0]
        recommended_songs.append(f"{song} by {artist_name}")
    
    return recommended_songs


if __name__ == '__main__':
    app.run(debug=True, port=5000)
