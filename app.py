from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import logging
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Model Loading ────────────────────────────────────────────────────────────
pipe = None

def load_model():
    global pipe
    try:
        from transformers import pipeline
        # ViT fine-tuned specifically for GAN / Deepfake detection
        # Model card: https://huggingface.co/Wvolf/ViT-L-16-for-GAN-and-Deepfake-Detection
        pipe = pipeline(
            "image-classification",
            model="Wvolf/ViT-L-16-for-GAN-and-Deepfake-Detection",
            device=-1  # CPU (Render free tier has no GPU)
        )
        logger.info("✅ GAN Detection model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        pipe = None


def detect_gan(image_base64: str):
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if pipe is None:
        return {"error": "Model not loaded"}, 503

    results = pipe(img)
    # Results are like: [{"label": "fake", "score": 0.92}, {"label": "real", "score": 0.08}]
    top = results[0]
    label = top["label"].lower()
    score = round(top["score"], 2)

    is_fake = "fake" in label or "gan" in label or "artificial" in label

    return {
        "is_fake": is_fake,
        "confidence": score
    }, 200


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "GAN & Deepfake Detection API",
        "model": "Wvolf/ViT-L-16-for-GAN-and-Deepfake-Detection",
        "version": "2.0.0"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": pipe is not None})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "Missing field: image_base64"}), 400

        try:
            img_bytes = base64.b64decode(data["image_base64"])
            if len(img_bytes) < 100:
                return jsonify({"error": "Image too small or invalid"}), 400
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

        result, status = detect_gan(data["image_base64"])
        return jsonify(result), status

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Startup ─────────────────────────────────────────────────────────────────
with app.app_context():
    load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
