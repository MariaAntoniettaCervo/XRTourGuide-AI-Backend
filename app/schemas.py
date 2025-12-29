from pydantic import BaseModel
from typing import List
from typing import Optional

    #--- TITOLI ---

class TitleRequest(BaseModel):
    original_title: str
    
class TitleResponse(BaseModel):
    original: str
    options: List[str] # Restituiremo 3 varianti
    best_option: str


    # --- DESCRIZIONI ---

class DescriptionRequest(BaseModel):
    original_text: str     
    target_lang: str = "it" # Per ora default italiano

class DescriptionResponse(BaseModel):
    full_text_optimized: str  # Testo intero da mostrare a schermo (UI)
    tts_chunks: List[str]     # Lista di frasi <180char per il motore audio

    # --- FIX MARKDOWN ---

class MarkdownFixRequest(BaseModel):
    text: str
    tone: str = "professional" # Opzionale: professional, friendly, academic

class MarkdownFixResponse(BaseModel):
    original_text: str
    fixed_text: str
    success: bool            # <--- NUOVO: True se l'AI ha risposto, False se errore
    error_message: Optional[str] = None # <--- NUOVO: Descrizione dell'errore    

    # --- AUDIO ON DEMAND ---
class AudioGenerationRequest(BaseModel):
    text: str          # La singola frase da leggere
    language: str = "it"

class AudioGenerationResponse(BaseModel):
    audio_url: str     # L'URL di MinIO da suonare
    cached: bool       # Debug: ci dice se era già pronto o no