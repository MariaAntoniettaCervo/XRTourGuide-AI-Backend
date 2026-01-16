import json
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # 1. Calcolo dinamico dei percorsi assoluti
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(base_dir)
        
        # Percorso assoluto per config.json
        self.config_file = os.path.join(self.project_root, "config.json")
        
        # Percorso assoluto per la voce di riferimento
        self.tts_voice_file = os.path.join(self.project_root, "app", "tts", "assets", "ref_voice.wav")

        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "tour-guide-assets")
        
        # Gestione booleana per secure url
        secure_env = os.getenv("SECURE_URL", "False")
        self.minio_secure = secure_env.lower() == "true"

        # Defaults
        self.tts_engine = "coqui-xtts"
        self.llm_model_name = "qwen2.5:7b"
        
        self.load()

    def load(self):
        """Carica le impostazioni dal file JSON se esiste"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.tts_engine = data.get("tts_engine", "coqui-xtts")
                    self.llm_model_name = data.get("llm_model_name", "qwen2.5:7b")
            except Exception as e:
                print(f"Errore caricamento config.json: {e}")

    def save(self):
        """Salva le impostazioni correnti su JSON"""
        data = {
            "tts_engine": self.tts_engine,
            "llm_model_name": self.llm_model_name
        }
        try:
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Errore salvataggio config.json: {e}")

global_settings = Settings()