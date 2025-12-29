import os
import torch
import torchaudio
import torchaudio.transforms as T
import gc 
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from app.interfaces.tts_interface import TTSInterface
from app.tts.text_normalizer import TextNormalizer
from app.utils.audio_converter import AudioConverter
from app.settings import global_settings  

class CoquiTTS(TTSInterface):
    # Parametri Audio
    GEN_SPEED = 1.1        
    PITCH_STEPS = -0.5     
    GPT_COND_LEN = 6       
    TEMP = 0.2             
    REP_PENALTY = 5.0      
    TOP_K = 60
    TOP_P = 0.8

    def __init__(self):
        print("IInizializzazione Plugin Coqui XTTS (Lazy Mode)...")
        
        self.normalizer = TextNormalizer()
        
        # Inizializzazione a None per risparmiare memoria
        self.model = None
        self.config = None

    def _load_model(self):
        """Carica il modello SOLO se non è già presente"""
        if self.model is not None:
            return

        print("Caricamento XTTS v2 in VRAM (Lazy Load)...")
        app_data = os.getenv('LOCALAPPDATA')
        model_path = os.path.join(app_data, "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
        
        if not os.path.exists(model_path):
            print("Scaricamento modello...")
            TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        self.config = XttsConfig()
        self.config.load_json(os.path.join(model_path, "config.json"))
        
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, eval=True)
        
        if torch.cuda.is_available():
            self.model.cuda()
            print("XTTS caricato su GPU")
        else:
            print("XTTS su CPU")

    def release_memory(self):
        """Scarica il modello per liberare spazio"""
        if self.model is not None:
            print("Rilascio VRAM XTTS...")
            self.model.cpu() 
            del self.model
            self.model = None
            self.config = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_audio(self, text: str, output_filename: str) -> bool:
        """Ciclo completo: Carica -> Genera -> Scarica"""
        
        # 1. CARICAMENTO FORZATO (Questo risolve l'errore 'NoneType')
        self._load_model()

        # Recuperiamo la voce dai settings dinamici
        # (Assicurati di aver aggiunto tts_voice_file in settings.py come stringa)
        voice_path = global_settings.tts_voice_file 
        
        if not os.path.exists(voice_path):
            print(f"Errore: File voce non trovato in {voice_path}")
            return False
        
        clean_text = self.normalizer.clean_text(text)
        is_mp3 = output_filename.endswith(".mp3")
        wav_temp = output_filename.replace(".mp3", ".wav") if is_mp3 else output_filename
        
        success = False
        
        try:
            # Controllo paranoico
            if self.model is None:
                raise Exception("CRITICO: Il modello è ancora None")

            # 2. Generazione
            with torch.no_grad():
                outputs = self.model.synthesize(
                    clean_text,
                    self.config,
                    speaker_wav=voice_path,
                    language="it",
                    gpt_cond_len=self.GPT_COND_LEN,
                    speed=self.GEN_SPEED,
                    temperature=self.TEMP,
                    repetition_penalty=self.REP_PENALTY,
                    top_k=self.TOP_K,
                    top_p=self.TOP_P,
                    enable_text_splitting=False,
                    do_sample=True
                )

                # 3. Post-Processing
                wav_tensor = torch.tensor(outputs["wav"]).unsqueeze(0)

                if self.PITCH_STEPS != 0:
                    pitch_shifter = T.PitchShift(sample_rate=24000, n_steps=self.PITCH_STEPS)
                    wav_tensor = pitch_shifter(wav_tensor)

                max_val = torch.abs(wav_tensor).max()
                if max_val > 0: wav_tensor = wav_tensor / max_val * 0.95

                resampler = T.Resample(orig_freq=24000, new_freq=44100, dtype=torch.float32)
                wav_hq = resampler(wav_tensor)
                
                torchaudio.save(wav_temp, wav_hq.detach().cpu(), 44100, bits_per_sample=32)

            # 4. Conversione
            if is_mp3:
                AudioConverter.convert_wav_to_mp3(wav_temp)
                success = os.path.exists(output_filename)
            else:
                success = True

        except Exception as e:
            print(f"Errore XTTS: {e}")
            if os.path.exists(wav_temp):
                try: os.remove(wav_temp)
                except: pass
            success = False
        
        finally:
            self.release_memory()
        
        return success