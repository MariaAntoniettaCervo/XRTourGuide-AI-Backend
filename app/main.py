from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import hashlib
import uvicorn

# Schemas
from app.schemas import (
    TitleRequest, TitleResponse, 
    DescriptionRequest, DescriptionResponse,
    AudioGenerationRequest, AudioGenerationResponse,
    MarkdownFixRequest, MarkdownFixResponse
)

# Factory
from app.tts.factory import TTSFactory

# Services
from app.llm.services.optimize_title import generate_optimized_title
from app.llm.services.optimize_description import generate_optimized_description
from app.llm.services.optimize_markdown import fix_text_logic

# Storage
from app.storage import check_file_exists, get_file_url, upload_file

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Avvio Backend XRTourGuide...")
    # Qui non serve piu inizializzare il motore TTS, viene gestito dalla Factory
    yield
    print("Shutdown Backend.")

app = FastAPI(title="XRTourGuide API", lifespan=lifespan)

# --- ENDPOINTS ---

@app.get("/")
def root():
    """Verifica se il server è vivo"""
    return {
        "status": "online", 
        "service": "XRTourGuide AI Backend", 
        "version": "2.0 (Modular)"
    }

@app.post("/generate-audio", response_model=AudioGenerationResponse)
async def generate_audio_ondemand(request: AudioGenerationRequest):
    """
    Chiamato dall'App del turista quando serve l'audio.
    Input: Una frase di testo.
    Output: URL MP3 (Generato al volo o recuperato da Cache MinIO).
    """
    
    # 1. Calcolo Hash (Impronta digitale della frase)
    text_hash = hashlib.md5(request.text.encode('utf-8')).hexdigest()
    
    # 2. Definizione Percorso MinIO (es:a1b2c3d4.mp3)
    object_name = f"{text_hash}.mp3"
    
    # 3. STRATEGIA CACHE: Controllo se esiste già
    if check_file_exists(object_name):
        print(f"CACHE HIT: {object_name}")
        return AudioGenerationResponse(
            audio_url=get_file_url(object_name),
            cached=True
        )
    
    # 4. GENERAZIONE (Se non esiste)
    print(f"NEW TTS: Generazione per '{request.text[:20]}...'")
    
    local_temp = f"temp_{text_hash}.mp3"
    
    try:
        # RECUPERO IL MOTORE DALLA FACTORY
        # Non usiamo variabili globali, istanziamo (o recuperiamo il singleton) qui
        tts_engine = TTSFactory.get_engine()
        
        # Generazione Audio
        success = tts_engine.generate_audio(request.text, local_temp)
        
        if not success or not os.path.exists(local_temp):
            raise Exception("Il motore TTS ha restituito False o il file non è stato creato.")
        
        # 5. Upload su MinIO
        upload_file(local_temp, object_name)
        
        # 6. Pulizia file locale temporaneo
        os.remove(local_temp)
        
        # Ritorna URL pubblico
        return AudioGenerationResponse(
            audio_url=get_file_url(object_name),
            cached=False
        )

    except Exception as e:
        print(f"ERRORE GENERAZIONE AUDIO: {e}")
        # Pulizia in caso di errore
        if os.path.exists(local_temp):
            os.remove(local_temp)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/title", response_model=TitleResponse)
async def optimize_title_endpoint(request: TitleRequest):
    """
        Riceve un titolo scritto dall'autore del tour e ne restituisce 3 varianti ottimizzate secondo i patter scelti
    """

    if not request.original_title:
        raise HTTPException(status_code=400, detail="Il titolo non può essere vuoto")
        
    # Chiamata al servizio LLM
    result = generate_optimized_title(request.original_title)
    
    return result

@app.post("/optimize/description", response_model=DescriptionResponse)
async def optimize_description_endpoint(request: DescriptionRequest):
    """
    Riceve una descrizione grezza e restituisce:
    1. Testo bello per la UI.
    2. Chunk audio pronti per XTTS.
    """
    if not request.original_text:
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto")
    
    # Chiamata al servizio
    result = generate_optimized_description(request.original_text)
    
    return result

@app.post("/optimize-markdown", response_model=MarkdownFixResponse)
async def fix_markdown_endpoint(request: MarkdownFixRequest):
    """
    Riceve un testo Markdown, se presenta errori lo corregge con l'AI 
    e restituisce la versione pulita.
    """
    # Chiama la logica LLM
    return fix_text_logic(request.text, request.tone)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)