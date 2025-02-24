from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Automatically handle CORS

# Load model and vectorizer
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
    
    # Manually add CORS headers to the response
    response = jsonify({"sentiment": int(prediction)})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    
    return response

# Handle OPTIONS preflight requests
@app.route("/analyze", methods=["OPTIONS"])
def handle_options():
    response = jsonify({"message": "CORS preflight handled"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
