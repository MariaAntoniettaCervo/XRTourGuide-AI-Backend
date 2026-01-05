from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import os
import asyncio
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
from app.llm.services.optimize_markdown import fix_markdown

# Storage
from app.storage import check_file_exists, get_file_url, upload_file, delete_file, save_json_to_minio, get_json_from_minio, init_storage

# --- GLOBAL LOCK PER TTS ---
# Assicura che avvenga solo UNA generazione pesante alla volta
tts_lock = asyncio.Lock()

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Avvio Backend XRTourGuide...")
    init_storage()
    
    yield
    print("Shutdown Backend.")

app = FastAPI(title="XRTourGuide API", lifespan=lifespan)

async def background_audio_task(text_to_read: str, object_name: str, local_temp: str):
    
    text_hash = hashlib.md5(text_to_read.encode('utf-8')).hexdigest()
    json_object_name = f"{text_hash}.json"
    error_object_name = f"{text_hash}.err"
    
    async with tts_lock:
        try:
            tts_engine = TTSFactory.get_engine()
            
            # 1. Recupero Dati 
            loaded_chunks = []
            
            if check_file_exists(json_object_name):
                print(f"WORKER: Trovato copione JSON {json_object_name}")
                data = get_json_from_minio(json_object_name)
                # Recuperiamo i chunks se esistono, altrimenti lista vuota
                loaded_chunks = data.get("tts_chunks", [])
            
            print(f"WORKER: Avvio Engine TTS...")
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                lambda: tts_engine.generate_audio(
                    text=text_to_read, 
                    output_filename=local_temp, 
                    chunks=loaded_chunks
                )
            )
            
            if success and os.path.exists(local_temp):
                upload_file(local_temp, object_name)
                try: delete_file(error_object_name)
                except: pass
                os.remove(local_temp)
            else:
                raise Exception("TTS Engine returned False.")

        except Exception as e:
            print(f"WORKER ERROR: {e}")
            # Creazione file .err
            local_err = local_temp.replace(".mp3", ".err")
            with open(local_err, "w") as f:
                f.write(str(e))
            upload_file(local_err, error_object_name)
            
            # Pulizia
            if os.path.exists(local_temp): os.remove(local_temp)
            if os.path.exists(local_err): os.remove(local_err)

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {
        "status": "online", 
        "service": "XRTourGuide AI Backend", 
        "version": "2.0 (Modular)"
    }

@app.post("/generate-audio", response_model=AudioGenerationResponse, status_code=202)
async def generate_audio(request: AudioGenerationRequest, background_tasks: BackgroundTasks):
    

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Testo vuoto")

    text_hash = hashlib.md5(request.text.encode('utf-8')).hexdigest()
    object_name = f"{text_hash}.mp3"
    error_object_name = f"{text_hash}.err"
    audio_url = get_file_url(object_name)


    if check_file_exists(object_name):
        return AudioGenerationResponse(
            audio_url=audio_url,
            status="ready",
            message="Audio pronto."
        )
    
    # 2. CONTROLLO ERRORE 
    if check_file_exists(error_object_name):
        # Se l'utente NON ha chiesto il retry esplicito, restituiamo l'errore
        if not request.retry:
            print(f"RITORNO ERRORE: Trovato {error_object_name}")
            return AudioGenerationResponse(
                audio_url="", # Nessun audio disponibile
                status="error",
                message="La generazione precedente è fallita. Invia 'retry': true per riprovare."
            )
        else:
            # Se l'utente HA chiesto retry, cancelliamo l'errore e procediamo
            print(f"RETRY RICHIESTO: Cancello {error_object_name} e riprovo.")
            try: delete_file(error_object_name)
            except: pass

    # 3. AVVIO GENERAZIONE (Nuova o Retry)
    local_temp = f"temp_{text_hash}.mp3"
    
    print(f"NEW REQUEST: Accodata generazione per '{request.text[:15]}...'")
    background_tasks.add_task(background_audio_task, request.text, object_name, local_temp)
    
    return AudioGenerationResponse(
        audio_url=audio_url,
        status="processing",
        message="Elaborazione in corso..."
    )


@app.post("/optimize/title", response_model=TitleResponse)
async def optimize_title_endpoint(request: TitleRequest):
    """
        Riceve un titolo scritto dall'autore del tour e ne restituisce 3 varianti ottimizzate secondo i patter scelti
    """

    if not request.original_title:
        raise HTTPException(status_code=400, detail="Il titolo non può essere vuoto")
        
    result = generate_optimized_title(request.original_title)
    
    return result

@app.post("/optimize/description", response_model=DescriptionResponse)
async def optimize_description_endpoint(request: DescriptionRequest):

    if not request.original_text:
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto")
    
    text_hash = hashlib.md5(request.original_text.encode('utf-8')).hexdigest()
    json_object_name = f"{text_hash}.json"
    
    result = generate_optimized_description(request.original_text)

    payload_to_save = {
        "full_text_optimized": result.full_text_optimized,
        "tts_chunks": result.tts_chunks
    }
    
    save_json_to_minio(payload_to_save, json_object_name)
    
    return result

@app.post("/optimize-markdown", response_model=MarkdownFixResponse)
async def fix_markdown_endpoint(request: MarkdownFixRequest):
    """
    Riceve un testo Markdown, se presenta errori lo corregge con l'AI 
    e restituisce la versione pulita.
    """
    return fix_markdown(request.text, request.tone)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)