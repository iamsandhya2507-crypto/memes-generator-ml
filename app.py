from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

CAPTIONS = [
    ("study", "Me opening the book and suddenly needing a snack."),
    ("study", "When the exam is tomorrow and the syllabus is infinite."),
    ("study", "I studied for five minutes. Time for a three-hour break."),
    ("coding", "It works perfectly until someone watches me run it."),
    ("coding", "One tiny bug, twenty browser tabs, zero confidence."),
    ("coding", "When the code finally runs and you don't touch anything."),
    ("work", "My productivity before coffee versus after coffee."),
    ("work", "When the meeting could have been one message."),
    ("friends", "Making plans together and cancelling together."),
    ("friends", "That one friend who says 'I'm five minutes away' from home."),
    ("everyday", "Me: I will sleep early. Also me at 2 AM:"),
    ("everyday", "When you clean your room and immediately cannot find anything."),
]

def recommend_captions(query, limit=4):
    query = query.strip() or "everyday life"
    documents = [topic + " " + caption for topic, caption in CAPTIONS]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(zip(scores, CAPTIONS), key=lambda item: item[0], reverse=True)
    return [{"topic": item[1][0], "text": item[1][1]} for item in ranked[:limit]]

@app.route("/", methods=["GET", "POST"])
def index():
    query, results = "", []
    if request.method == "POST":
        query = request.form.get("topic", "")
        results = recommend_captions(query)
    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
