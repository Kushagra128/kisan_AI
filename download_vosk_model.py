import os
import requests
import zipfile
import io

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip"
MODELS_DIR = os.path.join("k:", os.sep, "kisan_django", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "vosk-model-hi-0.22")

os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading Vosk Hindi model (approx 42MB)...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    print("Download complete. Extracting...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(MODELS_DIR)
    print("Extraction complete.")
else:
    print("Model already exists.")
