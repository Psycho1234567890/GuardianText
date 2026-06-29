from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Enable CORS for all origins
CORS(app, origins="*")

# Additional headers for browser preflight requests
@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response


# Load model
model = tf.keras.models.load_model(
    "simple_rnn_model.h5"
)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50


def generate_text(seed_text, next_words=20, temperature=0.8):

    result = seed_text

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [result]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=MAX_LEN,
            padding="pre"
        )

        prediction = model.predict(
            token_list,
            verbose=0
        )[0]

        prediction = np.log(
            prediction + 1e-8
        ) / temperature

        prediction = np.exp(prediction)
        prediction = prediction / np.sum(prediction)

        predicted_id = np.random.choice(
            len(prediction),
            p=prediction
        )

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                output_word = word
                break

        if output_word == "":
            break

        result += " " + output_word

    return result


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Neural Muse API Running"
    })


# Handle browser preflight request
@app.route("/generate", methods=["OPTIONS"])
def options_generate():
    return jsonify({"message": "OK"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        seed_text = data.get("text", "")
        temperature = float(
            data.get("temperature", 0.8)
        )

        word_count = int(
            data.get("word_count", 20)
        )

        output = generate_text(
            seed_text,
            word_count,
            temperature
        )

        return jsonify({
            "generated_text": output
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )