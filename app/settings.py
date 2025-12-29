import json
import os

class Settings:
    def __init__(self):
        # 1. Calcolo dinamico dei percorsi assoluti
        # Ottiene la cartella dove si trova questo file (app/)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Risale alla root del progetto (XRTourGuide-AI-Backend/)
        self.project_root = os.path.dirname(base_dir)
        
        # Percorso assoluto per config.json
        self.config_file = os.path.join(self.project_root, "config.json")
        
        # Percorso assoluto per la voce di riferimento
        # Risolve l'errore: "File voce non trovato"
        self.tts_voice_file = os.path.join(self.project_root, "app", "tts", "assets", "ref_voice.wav")

        # Defaults
        self.tts_engine = "coqui"
        self.llm_model_name = "qwen2.5:7b"
        
        self.load()

    def load(self):
        """Carica le impostazioni dal file JSON se esiste"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                data = json.load(f)
                self.tts_engine = data.get("tts_engine", "coqui")
                self.llm_model_name = data.get("llm_model_name", "qwen2.5:7b")

    def save(self):
        """Salva le impostazioni correnti su JSON"""
        data = {
            "tts_engine": self.tts_engine,
            "llm_model_name": self.llm_model_name
        }
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=4)

global_settings = Settings()