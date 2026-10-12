import os
import json
import time
from flask import Flask, request, jsonify
from gradio_client import Client
import boto3
from botocore.config import Config
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Existing Hugging Face Space configuration
HF_SPACE = os.environ.get("HF_SPACE")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Backblaze B2 credentials (kept secret as environment variables)
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")

# Static Backblaze B2 details directly inside the code
B2_BUCKET_NAME = "Derma-ai"
B2_ENDPOINT_URL = "https://s3.eu-central-003.backblazeb2.com"

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

        result = clean_response(str(result))
        return jsonify({"success": True, "result": result})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/crowdsource", methods=["POST"])
def crowdsource():
    # New crowdsourcing endpoint:
    # - Intercepts visual image file upload
    # - Saves to B2 bucket in 'images/'
    # - Appends/registers label metadata inside B2
    if not s3_client:
        return jsonify({
            "success": False,
            "error": "Backblaze B2 storage is not configured on the proxy server."
        }), 500

    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part in request"}), 400

        file = request.files['file']
        label = request.form.get('label', 'unlabeled').strip()

        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        # Standardize filenames safely
        timestamp = int(time.time())
        safe_file_name = secure_filename(file.filename)
        b2_filename = f"images/{timestamp}_{safe_file_name}"

        # 1. Upload the image directly to B2
        file_stream = file.stream
        s3_client.upload_fileobj(
            file_stream,
            B2_BUCKET_NAME,
            b2_filename,
            ExtraArgs={'ContentType': file.content_type or 'image/jpeg'}
        )

        # 2. Update metadata.json tracking file on B2 bucket
        metadata_key = "metadata.json"
        metadata_list = []

        # Try retrieving current metadata list if exists
        try:
            existing_meta_obj = s3_client.get_object(Bucket=B2_BUCKET_NAME, Key=metadata_key)
            metadata_list = json.loads(existing_meta_obj['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist yet, start fresh
            metadata_list = []
        except Exception as e:
            print(f"Warning retrieving metadata file: {e}")

        # Append current submission record
        metadata_list.append({
            "timestamp": timestamp,
            "file_path": b2_filename,
            "label": label
        })

        # Upload updated metadata list back to B2
        s3_client.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(metadata_list, indent=4),
            ContentType='application/json'
        )

        return jsonify({
            "success": True,
            "message": "Image and metadata successfully crowdsourced to Backblaze B2!",
            "file_path": b2_filename
        })

    except Exception as e:
        print(f"Crowdsource upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)