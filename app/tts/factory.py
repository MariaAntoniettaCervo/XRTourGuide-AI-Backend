from app.tts.plugins.coqui_engine import CoquiTTS    
from app.tts.plugins.piper_engine import PiperTTS    

class TTSFactory:
    _coqui_instance = None

    @staticmethod
    def get_engine(engine_type: str):
        if engine_type == "coqui-xtts":  
            if TTSFactory._coqui_instance is None:
                print("FACTORY: Caricamento modello Coqui in memoria...")
                TTSFactory._coqui_instance = CoquiTTS()
            return TTSFactory._coqui_instance
            
        elif engine_type == "piper":
            return PiperTTS()
            
        else:
            # Fallback o errore per engine sconosciuto
            raise ValueError(f"Engine TTS sconosciuto o non supportato: {engine_type}")