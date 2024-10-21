import random
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
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

nlp = spacy.load('en_core_web_sm')

songs_df = pd.read_csv('Content_based_Filtering.csv')
emotion_df = pd.read_csv('emotion_dataset.csv')
user_ratings_df = pd.read_csv('Updated_Collaborative_Filtering.csv')

indices = pd.Series(songs_df.index, index=songs_df['Song Name']).drop_duplicates()

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(songs_df['Mood']) 
cosine_sim = cosine_similarity(count_matrix)


emotion_df['Clean_text'] = emotion_df['Text'].apply(nfx.remove_stopwords)
emotion_df['Clean_text'] = emotion_df['Clean_text'].apply(nfx.remove_punctuations)
emotion_df['Clean_text'] = emotion_df['Clean_text'].apply(nfx.remove_userhandles)


emotion_df['Emotion'] = emotion_df['Emotion'].replace({
    'shame': 'sadness',
    'fear': 'sadness',
    'disgust': 'sadness',
    'surprise': 'joy'
})


Xfeatures = emotion_df['Clean_text']
ylabel = emotion_df['Emotion']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
X_train, X_test, y_train, y_test = train_test_split(X, ylabel, test_size=0.3, random_state=42)
nv_model = MultinomialNB()
nv_model.fit(X_train, y_train)


funsong = songs_df[songs_df['Mood'] == 'Fun']['Song Name'].tolist()
enersong = songs_df[songs_df['Mood'] == 'Energetic']['Song Name'].tolist()


def content_based_filtering(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    song_indices = [i[0] for i in sim_scores]
    return songs_df['Song Name'].iloc[song_indices].tolist()


def collaborative_filtering(song_name):
    user_ratings_df_numeric = user_ratings_df.drop(columns=['User ID'], errors='ignore')
    song_ratings = user_ratings_df_numeric[song_name]
    similar_songs = user_ratings_df_numeric.corrwith(song_ratings)
    corr_df = pd.DataFrame(similar_songs, columns=['Correlation']).dropna()
    sorted_songs = corr_df.sort_values(by='Correlation', ascending=False).index.tolist()
    return [song for song in sorted_songs if song != song_name][:1]


class InteractiveConversationalBot:
    def __init__(self):
        self.memory = defaultdict(list)
        self.user_name = None
        self.default_responses = [
            "Could you tell me more about that?",
            "That sounds interesting! What else?",
            "And then what happened?",
            "Do you often think about this?",
            "How does that make you feel?"
        ]

    def identify_names(self, text):
        doc = nlp(text)
        return {
            "People Names": list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])),
            "Organization Names": list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"])),
            "Location Names": list(set([ent.text for ent in doc.ents if ent.label_ == "GPE"]))
        }

    def generate_entity_based_question(self, entities):
        questions = []
        if entities["People Names"]:
            person = random.choice(entities["People Names"])
            questions.extend([f"How do you know {person}?", f"Is {person} someone important to you?"])
        if entities["Organization Names"]:
            org = random.choice(entities["Organization Names"])
            questions.extend([f"Have you worked with {org} before?", f"What do you think about {org}?"])
        if entities["Location Names"]:
            loc = random.choice(entities["Location Names"])
            questions.extend([f"Have you ever visited {loc}?", f"What do you like most about {loc}?"])
        return random.choice(questions) if questions else None

    def extract_keywords(self, text):
        rake = Rake()
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases() or []

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def personalize_response(self, text):
        sentiment = self.analyze_sentiment(text)
        if sentiment > 0.5:
            return "I'm glad to hear that!"
        elif sentiment < 0.5:
            return "I'm sorry to hear that. Do you want to talk more about it?"
        else:
            return random.choice(self.default_responses)

    def get_response(self, user_input):
        if user_input.lower() in ["bye", "quit", "exit"]:
            return "Thank you for chatting! Goodbye!"
        if user_input.lower() in ["hi", "hey", "hello"]:
            return "Hello! What's your name?"
        if not self.user_name:
            self.user_name = user_input
            return f"Nice to meet you, {self.user_name}! What would you like to talk about today?"

        entities = self.identify_names(user_input)
        if (question := self.generate_entity_based_question(entities)):
            return question

        keywords = self.extract_keywords(user_input)
        if keywords:
            keyword = random.choice(keywords)
            if keyword not in self.memory:
                self.memory[keyword].append(user_input)
                return f"{self.personalize_response(user_input)} {random.choice(self.default_responses)}"

        return random.choice(self.default_responses)

bot = InteractiveConversationalBot()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    chatbot_response = bot.get_response(user_input)
    emotion = predict_emotion(user_input)
    songs = recommend_songs(emotion)
    return jsonify({"chatbot_response": chatbot_response, "recommended_songs": songs})

def predict_emotion(text):
    vect = cv.transform([text]).toarray()
    return nv_model.predict(vect)[0]

def recommend_songs(emotion):
    if emotion == 'joy':
        selected_songs = random.sample(funsong, k=4)
    elif emotion == 'sadness':
        selected_songs = random.sample(enersong, k=4)
    else:
        selected_songs = random.sample(funsong + enersong, k=4)

    similar_song = collaborative_filtering(selected_songs[0])
    all_songs = selected_songs + similar_song
    return list(dict.fromkeys(all_songs)) 

if __name__ == '__main__':
    app.run(debug=True, port=5000)
