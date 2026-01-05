import json
from minio import Minio
from minio.error import S3Error
import io
from app.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, SECURE_URL

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=SECURE_URL
)

def init_storage():
    """Controlla la connessione e crea il bucket se necessario. Chiamare all'avvio dell'app."""
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            print(f"STORAGE: Creazione bucket '{MINIO_BUCKET}'...")
            client.make_bucket(MINIO_BUCKET)
        else:
            print(f"STORAGE: Bucket '{MINIO_BUCKET}' trovato.")
    except Exception as e:
        print(f"ATTENZIONE: Impossibile connettersi a MinIO. Verifica che sia attivo. Errore: {e}")

if not client.bucket_exists(MINIO_BUCKET):
    client.make_bucket(MINIO_BUCKET)

def check_file_exists(object_name: str) -> bool:
    try:
        client.stat_object(MINIO_BUCKET, object_name)
        return True
    except S3Error:
        return False

def get_file_url(object_name: str) -> str:
    return client.get_presigned_url("GET", MINIO_BUCKET, object_name)

def upload_file(file_path: str, object_name: str):
    content_type = "audio/mpeg" if file_path.endswith(".mp3") else "audio/wav"
    client.fput_object(MINIO_BUCKET, object_name, file_path, content_type=content_type)

def delete_file(object_name: str):
    try:
        client.remove_object(MINIO_BUCKET, object_name)
        print(f"MinIO: Cancellato {object_name}")
    except S3Error as e:
        print(f"MinIO Delete Error ({object_name}): {e}")    

def save_json_to_minio(data: dict, object_name: str):
    try:
        json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        data_stream = io.BytesIO(json_bytes)
        
        client.put_object(
            MINIO_BUCKET,
            object_name,
            data_stream,
            length=len(json_bytes),
            content_type="application/json"
        )
        print(f"MinIO: Salvato JSON in memoria -> {object_name}")
        
    except Exception as e:
        print(f"Errore salvataggio JSON MinIO: {e}")

def get_json_from_minio(object_name: str) -> dict:

    response = None
    try:
        response = client.get_object(MINIO_BUCKET, object_name)
        content = response.read()
        return json.loads(content)
    except Exception as e:
        print(f"MinIO: Impossibile leggere JSON {object_name}: {e}")
        return {}
    finally:
        # È buona norma chiudere la connessione e rilasciarla al pool
        if response:
            response.close()
            response.release_conn()