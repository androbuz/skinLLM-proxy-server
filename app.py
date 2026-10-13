import os
import json
import time
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client
import boto3
from botocore.config import Config
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Existing Hugging Face Space configuration
HF_SPACE = os.environ.get("HF_SPACE")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Backblaze B2 credentials (kept secret as environment variables)
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")

# Static Backblaze B2 details directly inside the code
B2_BUCKET_NAME = "Derma-ai"
B2_ENDPOINT_URL = "https://s3.eu-central-003.backblazeb2.com"
B2_REGION = "eu-central-003"
B2_APK_KEY = "downloads/app.apk"

# Local cache path for the APK file
LOCAL_APK_PATH = os.path.join(tempfile.gettempdir(), "skinllm_app_cached.apk")

# Initialize Gradio client if HF credentials are provided
client = None
if HF_SPACE and HF_TOKEN:
    client = Client(HF_SPACE, token=HF_TOKEN, verbose=False)

# Initialize B2 S3 Client if credentials are provided
s3_client = None
if B2_KEY_ID and B2_APPLICATION_KEY:
    s3_client = boto3.client(
        's3',
        endpoint_url=B2_ENDPOINT_URL,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APPLICATION_KEY,
        config=Config(signature_version='s3v4')
    )

def download_apk_from_b2():
    '''Helper function to download the latest APK from B2 to the local cache.'''
    if not s3_client:
        raise Exception("B2 S3 Client is not initialized.")
    print("Downloading APK from B2 to cache...")
    s3_client.download_file(B2_BUCKET_NAME, B2_APK_KEY, LOCAL_APK_PATH)
    print("APK cached successfully.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "b2_configured": s3_client is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Original LLM text-based prediction endpoint
    if not client:
        return jsonify({"success": False, "error": "HF Client not initialized"}), 500
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "Request body must be JSON"}), 400

        user_input = data.get("user_input")
        if not user_input:
            return jsonify({"success": False, "error": "Missing user_input"}), 400

        result = client.predict(user_input, api_name="/predict")

        if isinstance(result, (tuple, list)):
            result = result[0]

        if isinstance(result, str):
            result = clean_response(result)

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
    text = text.strip()
    if "ANSWER:" in text:
        text = text.split("ANSWER:", 1)[1].strip()

    banned_blocks = [
        "IMAGE ANALYSIS RESULTS:",
        "Your task:",
        "<|user|>",
        "<|system|>",
        "<|assistant|"
    ]
    for block in banned_blocks:
        text = text.replace(block, "")
    return text.strip()

@app.route("/predict_visual", methods=["POST"])
def predict_visual():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        description = request.form.get('description', '')

        # Prepare the file data and fields to forward to Hugging Face
        files = {
            'file': (file.filename, file.read(), file.content_type or 'image/jpeg')
        }
        data = {
            'description': description
        }

        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        # Send standard multipart POST request directly to the Hugging Face space endpoint
        hf_url = "https://andro777-dermatology-ai.hf.space/predict"
        import requests
        response = requests.post(hf_url, files=files, data=data, headers=headers)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "success": False,
                "error": f"HF Space returned status code {response.status_code}",
                "details": response.text
            }), response.status_code

    except Exception as e:
        print(f"Visual prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# download cached apk route
@app.route("/download", methods=["GET"])
def download():
    try:
        # If not cached yet, download it once
        if not os.path.exists(LOCAL_APK_PATH):
            download_apk_from_b2()

        return send_file(
            LOCAL_APK_PATH,
            mimetype="application/vnd.android.package-archive",
            as_attachment=True,
            download_name="app.apk"
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to serve APK: {str(e)}"
        }), 500

# endpoint to trigger a fresh download/reload of the APK
@app.route("/reload-apk", methods=["GET", "POST"])
def reload_apk():
    try:
        download_apk_from_b2()
        return jsonify({
            "success": True,
            "message": "APK cache successfully updated from B2."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to update cache: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
