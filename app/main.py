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
    MarkdownFixRequest, MarkdownFixResponse,
    TTSModelEnum # Importiamo l'Enum
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
tts_lock = asyncio.Lock()

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Avvio Backend XRTourGuide...")
    init_storage()
    yield
    print("Shutdown Backend.")

app = FastAPI(title="XRTourGuide API", lifespan=lifespan)

# --- WORKER TASK ---

async def background_audio_task(text_to_read: str, object_name: str, local_temp: str, engine_name: str):
       
    json_object_name = f"{hashlib.md5(text_to_read.encode('utf-8')).hexdigest()}.json"
    error_object_name = object_name.replace(".mp3", ".err")
    
    async with tts_lock:
        try:

            print(f"WORKER: Richiesto engine {engine_name}")
            tts_engine = TTSFactory.get_engine(engine_name)
            
            # Recupero chunks salvati (se esistono)
            loaded_chunks = []
            if check_file_exists(json_object_name):
                print(f"WORKER: Trovato copione JSON {json_object_name}")
                data = get_json_from_minio(json_object_name)
                loaded_chunks = data.get("tts_chunks", [])
            
            print(f"WORKER: Avvio Generazione...")
            
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
            local_err = local_temp.replace(".mp3", ".err")
            with open(local_err, "w") as f:
                f.write(str(e))
            upload_file(local_err, error_object_name)
            
            if os.path.exists(local_temp): os.remove(local_temp)
            if os.path.exists(local_err): os.remove(local_err)

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {
        "status": "online", 
        "service": "XRTourGuide AI Backend", 
        "version": "2.1 (Multi-Model Support)"
    }

@app.post("/generate-audio", response_model=AudioGenerationResponse, status_code=202)
async def generate_audio(request: AudioGenerationRequest, background_tasks: BackgroundTasks):

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    # Calcolo Hash unico basato su testo + motore
    combo_string = f"{request.text}_{request.tts_engine.value}"
    text_hash = hashlib.md5(combo_string.encode('utf-8')).hexdigest()
    
    object_name = f"{text_hash}.mp3"
    error_object_name = f"{text_hash}.err"
    audio_url = get_file_url(object_name)

    if check_file_exists(object_name):
        return AudioGenerationResponse(
            audio_url=audio_url,
            status="ready",
            message=f"Audio già pronto (generato con {request.tts_engine.value})."
        )
    
    # Gestione Retry ed Errori
    if check_file_exists(error_object_name):
        if not request.retry:
            return AudioGenerationResponse(
                audio_url="",
                status="error",
                message="Errore precedente. Invia 'retry': true."
            )
        else:
            try: delete_file(error_object_name)
            except: pass

    # Avvio Task
    local_temp = f"temp_{text_hash}.mp3"
    print(f"NEW REQUEST: Audio '{request.tts_engine.value}' per '{request.text[:15]}...'")
    
    background_tasks.add_task(
        background_audio_task, 
        request.text, 
        object_name, 
        local_temp, 
        request.tts_engine.value
    )
    
    return AudioGenerationResponse(
        audio_url=audio_url,
        status="processing",
        message="Elaborazione in corso..."
    )


@app.post("/optimize/title", response_model=TitleResponse)
async def optimize_title_endpoint(request: TitleRequest):
    if not request.original_title:
        raise HTTPException(status_code=400, detail="Il titolo non può essere vuoto")
        
    result = generate_optimized_title(request.original_title, model_name=request.model.value)
    
    if not hasattr(result, "model_used"):
         pass 

    return result

@app.post("/optimize/description", response_model=DescriptionResponse)
async def optimize_description_endpoint(request: DescriptionRequest):
    if not request.original_text:
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto")
    
    result = generate_optimized_description(request.original_text, model_name=request.model.value)

    # Salvataggio chunks 
    text_hash = hashlib.md5(request.original_text.encode('utf-8')).hexdigest()
    json_object_name = f"{text_hash}.json"
    
    payload_to_save = {
        "full_text_optimized": result.full_text_optimized,
        "tts_chunks": result.tts_chunks
    }
    save_json_to_minio(payload_to_save, json_object_name)
    
    return result

@app.post("/optimize-markdown", response_model=MarkdownFixResponse)
async def fix_markdown_endpoint(request: MarkdownFixRequest):
    return fix_markdown(request.text, request.tone, model_name=request.model.value)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)