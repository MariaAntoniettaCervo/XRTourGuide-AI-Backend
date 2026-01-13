from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class LLMModelEnum(str, Enum):
    # Opzione Veloce / Balanced
    QWEN_7B = "qwen2.5:7b"
    # Opzione Qualità / Più Creativa
    LLAMA_8B = "llama3.1:8b"

class TTSModelEnum(str, Enum):
    # Opzione Veloce (CPU friendly)
    FAST_CPU = "piper" 
    # Opzione Qualità (GPU required, più lento ma realistico)
    QUALITY_GPU = "coqui-xtts"

# --- TITOLI ---

class TitleRequest(BaseModel):
    original_title: str = Field(..., example="Il Colosseo Romano")
    # Aggiungiamo la scelta del modello
    model: LLMModelEnum = Field(
        default=LLMModelEnum.QWEN_7B, 
        description="Scegli il modello AI per generare i titoli."
    )
    
class TitleResponse(BaseModel):
    original: str
    options: List[str] 
    best_option: str
    model_used: str 

# --- DESCRIZIONI ---

class DescriptionRequest(BaseModel):
    original_text: str = Field(..., example="Testo lungo da riassumere...")
    target_lang: str = "it"
    model: LLMModelEnum = Field(
        default=LLMModelEnum.QWEN_7B, 
        description="Scegli il modello AI per ottimizzare il testo."
    )

class DescriptionResponse(BaseModel):
    full_text_optimized: str  
    tts_chunks: List[str]     
    model_used: str

# --- FIX MARKDOWN ---

class MarkdownFixRequest(BaseModel):
    text: str
    tone: str = "professional" 
    model: LLMModelEnum = Field(default=LLMModelEnum.QWEN_7B)

class MarkdownFixResponse(BaseModel):
    original_text: str
    fixed_text: str
    success: bool           
    error_message: Optional[str] = None   

# --- AUDIO ---

class AudioGenerationRequest(BaseModel):
    text: str = Field(..., example="Ciao, benvenuti al tour!")
    retry: bool = False
    
    # Scelta del motore Audio
    tts_engine: TTSModelEnum = Field(
        default=TTSModelEnum.QUALITY_GPU,
        description="Scegli 'piper' per velocità o 'coqui-xtts' per qualità."
    )

class AudioGenerationResponse(BaseModel):
    audio_url: str
    status: str 
    message: Optional[str] = None