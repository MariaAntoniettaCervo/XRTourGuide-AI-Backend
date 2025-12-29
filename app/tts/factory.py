from app.settings import global_settings
from app.tts.plugins.coqui_engine import CoquiTTS   
from app.tts.plugins.piper_engine import PiperTTS   

class TTSFactory:
    _coqui_instance = None

    @staticmethod
    def get_engine():
        engine_type = global_settings.tts_engine
        
        if engine_type == "coqui":
            # Singleton per Coqui (pesante da caricare)
            if TTSFactory._coqui_instance is None:
                TTSFactory._coqui_instance = CoquiTTS()
            return TTSFactory._coqui_instance
            
        elif engine_type == "piper":
            return PiperTTS()
            
        raise ValueError(f"Engine TTS sconosciuto: {engine_type}")