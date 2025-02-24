from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text")
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return jsonify({"sentiment": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
