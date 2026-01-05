import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_VOICE_PATH = os.path.join(BASE_DIR, "tts", "assets", "ref_voice.wav")

# --- MINIO CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"   
MINIO_SECRET_KEY = "minioadmin"   
MINIO_BUCKET = "audio-tours"
SECURE_URL = False