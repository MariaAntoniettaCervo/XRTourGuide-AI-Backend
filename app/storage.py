from minio import Minio
from minio.error import S3Error
import json
import io
import os

from app.app_config import global_settings

# Inizializzazione Client
client = Minio(
    global_settings.minio_endpoint,
    access_key=global_settings.minio_access_key,
    secret_key=global_settings.minio_secret_key,
    secure=global_settings.minio_secure
)

def init_storage():
    """Controlla la connessione e imposta i permessi pubblici."""
    bucket_name = global_settings.minio_bucket
    try:
        # 1. Crea il bucket se non esiste
        if not client.bucket_exists(bucket_name):
            print(f"STORAGE: Creazione bucket '{bucket_name}'...")
            client.make_bucket(bucket_name)
        else:
            print(f"STORAGE: Bucket '{bucket_name}' trovato.")

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicRead",
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                }
            ]
        }
        client.set_bucket_policy(bucket_name, json.dumps(policy))
        print(f"STORAGE: Policy 'Public Read' applicata a '{bucket_name}'.")
            
    except Exception as e:
        print(f"ATTENZIONE: Errore configurazione MinIO: {e}")

def check_file_exists(object_name: str) -> bool:
    try:
        client.stat_object(global_settings.minio_bucket, object_name)
        return True
    except S3Error:
        return False

def get_file_url(object_name: str) -> str:

    protocol = "https" if global_settings.minio_secure else "http"
    return f"{protocol}://{global_settings.minio_endpoint}/{global_settings.minio_bucket}/{object_name}"

def upload_file(file_path: str, object_name: str):
    content_type = "audio/mpeg" if file_path.endswith(".mp3") else "audio/wav"
    try:
        client.fput_object(
            global_settings.minio_bucket, 
            object_name, 
            file_path, 
            content_type=content_type
        )
        print(f"MinIO: Upload completato -> {object_name}")
    except S3Error as e:
        print(f"MinIO Upload Error: {e}")
        raise e

def delete_file(object_name: str):
    try:
        client.remove_object(global_settings.minio_bucket, object_name)
        print(f"MinIO: Cancellato {object_name}")
    except S3Error as e:
        print(f"MinIO Delete Error ({object_name}): {e}")    

def save_json_to_minio(data: dict, object_name: str):
    try:
        json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
        data_stream = io.BytesIO(json_bytes)
        
        client.put_object(
            global_settings.minio_bucket,
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
        response = client.get_object(global_settings.minio_bucket, object_name)
        content = response.read()
        return json.loads(content)
    except Exception as e:
        print(f"MinIO: Impossibile leggere JSON {object_name}: {e}")
        return {}
    finally:
        if response:
            response.close()
            response.release_conn()