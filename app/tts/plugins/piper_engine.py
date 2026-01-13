import subprocess
import os
import sys
import importlib.util
from app.interfaces.tts_interface import TTSInterface
from app.tts.text_normalizer import TextNormalizer

class PiperTTS(TTSInterface):
    def __init__(self):
        print("Inizializzazione Plugin Piper TTS...")
        
        # --- AUTO-INSTALLAZIONE LIBRERIA ---
        self._ensure_piper_installed()

        # 1. Calcolo Percorsi Assoluti
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        tts_dir = os.path.dirname(current_dir)                   
        app_dir = os.path.dirname(tts_dir)                       
        self.project_root = os.path.dirname(app_dir)             
        
        # 2. Percorso Modello (Piper)
        self.model_path = os.path.join(self.project_root, "models", "it_IT-paola-medium.onnx")
        
        if not os.path.exists(self.model_path):
            print(f"ATTENZIONE: Modello Piper non trovato in: {self.model_path}")
            print("   -> Scaricalo e mettilo nella cartella 'models/'.")

        # 3. Percorso FFmpeg
        self.ffmpeg_path = os.path.join(self.project_root, "bin", "ffmpeg.exe")
        
        # Fallback sistema
        if not os.path.exists(self.ffmpeg_path):
            self.ffmpeg_path = "ffmpeg"

        self.normalizer = TextNormalizer()

    def _ensure_piper_installed(self):
        """
        Controlla se la libreria 'piper' è installata.
        Se manca, la installa automaticamente tramite pip.
        """
        spec = importlib.util.find_spec("piper")
        
        if spec is None:
            print("Libreria 'piper-tts' mancante. Installazione automatica in corso...")
            try:
                # Esegue 'pip install piper-tts' usando lo stesso interprete Python attivo
                subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
                print("Libreria Piper installata con successo!")
            except subprocess.CalledProcessError as e:
                print(f"ERRORE CRITICO: Impossibile installare piper-tts automaticamente.\nEsegui manualmente: pip install piper-tts\nDettagli: {e}")

    def generate_audio(self, text: str, output_filename: str, chunks: list = None, **kwargs) -> bool:
        """
        Implementazione del metodo obbligatorio dell'interfaccia TTSInterface.
        """
        
        # --- NORMALIZZAZIONE ---
        testo_processato = self.normalizer.clean_text(text)
        
        is_mp3 = output_filename.endswith(".mp3")
        # Se è mp3, creiamo prima un wav temporaneo
        wav_temp = output_filename.replace(".mp3", ".wav") if is_mp3 else output_filename
        
        # Comando per eseguire piper come modulo
        comando_piper = [
            sys.executable, "-m", "piper",
            "--model", self.model_path,
            "--output_file", wav_temp,
            "--sentence_silence", "0.5"
        ]

        # Configurazione ambiente per UTF-8 (importante per Windows)
        my_env = os.environ.copy()
        my_env["PYTHONUTF8"] = "1"
        my_env["PYTHONIOENCODING"] = "utf-8"

        try:
            # 1. Genera WAV con Piper
            subprocess.run(
                comando_piper, 
                input=testo_processato.encode('utf-8'),
                env=my_env,
                check=True,
                stdout=subprocess.DEVNULL, # Silenzia output standard
                stderr=subprocess.PIPE     # Cattura errori
            )
            
            # Se l'output richiesto era WAV, abbiamo finito
            if not is_mp3:
                return os.path.exists(wav_temp)

            # 2. Conversione MP3 con FFmpeg
            if os.path.exists(wav_temp):
                comando_ffmpeg = [
                    self.ffmpeg_path,
                    "-i", wav_temp,
                    "-y",                 # Sovrascrivi senza chiedere
                    "-b:a", "128k",       # Bitrate
                    "-af", "adelay=200|200", # Aggiunge 200ms di silenzio all'inizio
                    output_filename
                ]
                
                subprocess.run(
                    comando_ffmpeg,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # --- PULIZIA ---
                try:
                    os.remove(wav_temp)
                except Exception as e:
                    print(f"Impossibile rimuovere temp wav: {e}")
                
                if os.path.exists(output_filename):
                    return output_filename
            
            raise Exception("Conversione MP3 fallita")

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode('utf-8', errors='ignore')
            print(f"Errore Processo Piper/FFmpeg: {err_msg}")
            raise e # Rilanciamo l'errore per farlo vedere al worker
            
        except Exception as e:
            print(f"Errore Generico Piper Plugin: {e}")
            raise e