import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_VOICE_PATH = os.path.join(BASE_DIR, "tts", "assets", "ref_voice.wav")

# --- MINIO CONFIG ---
class Config:
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    
   
    MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "audio-tours")
    
    SECURE_URL = os.getenv("MINIO_SECURE", "False").lower() == "true"

