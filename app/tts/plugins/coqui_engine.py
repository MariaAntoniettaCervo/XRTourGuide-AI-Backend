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
from app.app_config import global_settings
from app.utils.text_processing import smart_chunking

class CoquiTTS(TTSInterface):
    # Parametri Audio
    GEN_SPEED = 1.1        
    PITCH_STEPS = -0.5     
    GPT_COND_LEN = 6       
    TEMP = 0.2             
    REP_PENALTY = 5.0      
    TOP_K = 60
    TOP_P = 0.8
    
    SENTENCE_PAUSE_S = 0.4 

    def __init__(self):
        print("Inizializzazione Plugin Coqui XTTS (Lazy Mode)...")
        self.normalizer = TextNormalizer()
        self.model = None
        self.config = None

    def _load_model(self):
        if self.model is not None: return

        print("Caricamento XTTS v2 in VRAM (Lazy Load)...")
        app_data = os.getenv('LOCALAPPDATA')
        if not app_data: app_data = "/root/.local/share" 
        
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
        if self.model is not None:
            print("Rilascio VRAM XTTS...")
            self.model.cpu() 
            del self.model
            self.model = None
            self.config = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_audio(self, text: str, output_filename: str, chunks: list[str] = None) -> bool:
        """Genera audio gestendo i chunks per evitare limiti di caratteri"""
        
        # 1. Logica Fallback Chunks
        chunks_to_use = chunks
        if not chunks_to_use:
            print("CoquiTTS: Chunks non forniti, calcolo locale in corso...")
            chunks_to_use = smart_chunking(text)
            
        if not chunks_to_use:
            print("Errore: Nessun testo da elaborare")
            return False

        self._load_model()

        voice_path = global_settings.tts_voice_file 
        if not os.path.exists(voice_path):
            print(f"Errore: File voce non trovato in {voice_path}")
            return False
        
        is_mp3 = output_filename.endswith(".mp3")
        wav_temp = output_filename.replace(".mp3", ".wav") if is_mp3 else output_filename
        
        generated_tensors = []
        sample_rate = 24000 

        try:
            with torch.no_grad():
                silence_frames = int(sample_rate * self.SENTENCE_PAUSE_S)
                silence_tensor = torch.zeros(1, silence_frames)
                if torch.cuda.is_available():
                    silence_tensor = silence_tensor.cuda()

                print(f"XTTS: Elaborazione di {len(chunks_to_use)} chunks...")

                for i, text_chunk in enumerate(chunks_to_use):
                    clean_text = self.normalizer.clean_text(text_chunk)
                    if not clean_text.strip(): continue

                    # Genera il singolo pezzo
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
                    
                    wav_tensor = torch.tensor(outputs["wav"]).unsqueeze(0)
                    if torch.cuda.is_available():
                        wav_tensor = wav_tensor.cuda()
                    
                    generated_tensors.append(wav_tensor)
                    
                    if i < len(chunks_to_use) - 1:
                        generated_tensors.append(silence_tensor)

                if not generated_tensors:
                    raise Exception("Nessun audio generato (chunks vuoti?)")

                # 2. UNIONE (Concatenazione dei pezzi)
                full_audio = torch.cat(generated_tensors, dim=1)

                # 3. Post-Processing (Sposta su CPU)
                full_audio = full_audio.cpu()

                if self.PITCH_STEPS != 0:
                    pitch_shifter = T.PitchShift(sample_rate=sample_rate, n_steps=self.PITCH_STEPS)
                    full_audio = pitch_shifter(full_audio)

                max_val = torch.abs(full_audio).max()
                if max_val > 0: full_audio = full_audio / max_val * 0.95

                resampler = T.Resample(orig_freq=sample_rate, new_freq=44100, dtype=torch.float32)
                wav_hq = resampler(full_audio)
                
                torchaudio.save(wav_temp, wav_hq, 44100, bits_per_sample=32)

            # 4. Conversione
            if is_mp3:
                AudioConverter.convert_wav_to_mp3(wav_temp)
                success = os.path.exists(output_filename)
            else:
                success = True

        except Exception as e:
            print(f"Errore XTTS Multi-Chunk: {e}")
            if os.path.exists(wav_temp):
                try: os.remove(wav_temp)
                except: pass
            success = False
        
        finally:
            self.release_memory()
        
        return success