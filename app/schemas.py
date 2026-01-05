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
    target_lang: str = "it" 

class DescriptionResponse(BaseModel):
    full_text_optimized: str  
    tts_chunks: List[str]     

    # --- FIX MARKDOWN ---

class MarkdownFixRequest(BaseModel):
    text: str
    tone: str = "professional" # Opzionale: professional, friendly, academic

class MarkdownFixResponse(BaseModel):
    original_text: str
    fixed_text: str
    success: bool           
    error_message: Optional[str] = None   

    # --- AUDIO ---
class AudioGenerationRequest(BaseModel):
    text: str
    retry: bool = False  # Se true, ignora errori precedenti e riprova

class AudioGenerationResponse(BaseModel):
    audio_url: str
    status: str  # "ready", "processing", "error"
    message: Optional[str] = None