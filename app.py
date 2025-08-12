from flask import Flask, request, jsonify, send_from_directory
import joblib
import os

import joblib

# After training

app = Flask(__name__, static_folder='static')

model = joblib.load("model.pkl")  # Make sure model.pkl is in root folder
# joblib.dump(model, "model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True)