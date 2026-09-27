
import os

from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Hugging Face Space configuration
HF_SPACE = os.environ["HF_SPACE"]
HF_TOKEN = os.environ["HF_TOKEN"]

# Connect to the private HF Space
client = Client(
    HF_SPACE,
    token=HF_TOKEN,
    verbose=False
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be JSON"
            }), 400

        user_input = data.get("user_input")

        if not user_input:
            return jsonify({
                "success": False,
                "error": "Missing user_input"
            }), 400

        # Call the private Hugging Face Space
        result = client.predict(
            user_input,
            api_name="/predict"
        )

        # Gradio can return tuples/lists depending on the interface
        if isinstance(result, (tuple, list)):
            result = result[0]

        # Clean the model output
        result = clean_response(str(result))

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        print(f"Prediction error: {e}")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def clean_response(text):
    """
    Clean the response returned by the HF Space.
    """

    text = text.strip()

    # If the model returns an ANSWER section,
    # keep everything after it.
    if "ANSWER:" in text:
        text = text.split("ANSWER:", 1)[1].strip()

    # Remove leaked prompt sections if present
    banned_blocks = [
        "IMAGE ANALYSIS RESULTS:",
        "Your task:",
        "<|user|>",
        "<|system|>",
        "<|assistant|>"
    ]

    for block in banned_blocks:
        text = text.replace(block, "")

    return text.strip()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
