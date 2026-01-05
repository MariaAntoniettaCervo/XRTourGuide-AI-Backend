import requests
import time
import json
import csv
import re
import math
import os

# --- CONFIGURAZIONE ---
MODELS = [
    "qwen2.5:7b",      
    "llama3.1:8b",     
    "mistral-nemo",
    "phi3.5"
]

OUTPUT_FILE = "benchmark_v3_strict.csv"
OLLAMA_API = "http://localhost:11434/api"

# Pesi per il calcolo del punteggio finale
# Nota: json_validity e gestito separatamente come penalita critica
WEIGHTS = {
    "tts_safety": 1.5,    
    "fidelity": 2.0,      
    "speed": 1.5,         
    "length_accuracy": 0.5, 
    "lexical_variety": 0.5, 
    "memory_load": -1.0   
}

# Penalita bloccante per fallimento JSON
PENALTY_BROKEN_JSON = -1000.0

# --- PROMPT (INVARIATO) ---
SYSTEM_PROMPT = """
### PATTERN: PERSONA
Agisci come un esperto di storia dell'arte empatico e carismatico.
Il tuo compito è revisionare ed espandere la descrizione fornita dall'utente.
Non sei un'enciclopedia, sei un narratore che accompagna il visitatore.

### PATTERN: AUDIENCE
Il tuo pubblico è composto da turisti generici, famiglie e curiosi. 
Evita il gergo accademico complesso. Usa un linguaggio caldo e accessibile.

### PATTERN: CONTEXT MANAGER
Rimani strettamente focalizzato sul monumento descritto in input.
**Mantieni le informazioni fattuali corrette fornite in input, ma rendile più avvincenti.**

### PATTERN: TEMPLATE & OUTPUT AUTOMATION
Devi rispondere ESCLUSIVAMENTE con un oggetto JSON valido. 
Non aggiungere saluti, premesse o testo fuori dal JSON.
Struttura richiesta:
{
  "titolo": "Un titolo accattivante basato sul testo (max 60 caratteri)",
  "descrizione_audio": "Il testo narrativo revisionato...",
  "fact_check": ["Lista array di date, nomi o fatti citati per verifica"]
}

### LUNGHEZZA OUTPUT
L'obiettivo è una durata di circa 1 minuto di parlato (tra le 130 e le 160 parole).
Se l'input è troppo breve, espandilo con dettagli sensoriali o storici pertinenti.
Se l'input è troppo lungo, sintetizzalo mantenendo i punti chiave.

### PUNCTUATION ENGINEERING (Regia Audio per TTS)
Scrivi il campo 'descrizione_audio' ottimizzato ESPLICITAMENTE per la lettura vocale:
1. Usa frequentemente i tre puntini (...) per indicare pause di respiro e suspense.
2. Usa punti esclamativi (!) per enfatizzare le emozioni.
3. Spezza le frasi lunghe in frasi brevi.
4. NON usare mai elenchi puntati, parentesi o numeri (es. "1200" -> "milleduecento").

"""

USER_INPUT = "Il Colosseo è un anfiteatro romano del primo secolo. Era usato per i giochi dei gladiatori ed è molto grande."

# --- ALGORITMI DI ANALISI ---

def get_next_run_id():
    """Legge il CSV per trovare l'ultimo Run_ID e lo incrementa."""
    if not os.path.isfile(OUTPUT_FILE):
        return 1
    
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            ids = [int(row.get('Run_ID', 0)) for row in reader if row.get('Run_ID', '').isdigit()]
            if not ids: return 1
            return max(ids) + 1
    except Exception as e:
        print(f"Attenzione: Impossibile leggere ID precedente ({e}). Ricomincio da 1.")
        return 1

def check_tts_safety(text):
    errors = []
    if re.search(r'\d', text):
        numbers = re.findall(r'\d+', text)
        errors.append(f"NUMBERS({len(numbers)})")
    if re.search(r'[%$@#\/]', text):
        errors.append("SYMBOLS")
    if re.search(r'[\(\)\[\]]', text):
        errors.append("PARENTHESES")

    score = max(0, 100 - (len(errors) * 25))
    return score, errors

def check_length_accuracy(text, min_w=100, max_w=150):
    words = len(text.split())
    if min_w <= words <= max_w:
        return 100, words
    
    if words < min_w:
        diff = min_w - words
    else:
        diff = words - max_w
        
    score = max(0, 100 - (diff * 2)) 
    return score, words

def calculate_ttr(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    if not words: return 0
    unique_words = set(words)
    return round(len(unique_words) / len(words), 2)

def calculate_fidelity(user_input, generated_text):
    clean_in = re.sub(r'[^\w\s]', '', user_input.lower())
    input_keys = {w for w in clean_in.split() if len(w) > 3 and w not in ["della", "sono", "questo"]}
    
    if not input_keys: return 100, []
    
    gen_lower = generated_text.lower()
    matches = 0
    missing = []
    
    for k in input_keys:
        if k[:-1] in gen_lower: matches += 1
        else: missing.append(k)
            
    score = (matches / len(input_keys)) * 100
    return round(score, 1), missing

def get_model_details(model_name):
    try:
        res = requests.post(f"{OLLAMA_API}/show", json={"name": model_name}).json()
        details = res.get("details", {})
        
        size_gb = 0
        try:
            tags = requests.get(f"{OLLAMA_API}/tags").json()
            for m in tags['models']:
                if m['name'] == model_name or m['name'] == model_name + ":latest":
                    size_gb = round(m['size'] / (1024**3), 2)
                    break
        except: pass

        return {
            "params": details.get("parameter_size", "N/A"),
            "quant": details.get("quantization_level", "N/A"),
            "family": details.get("family", "N/A"),
            "size_gb": size_gb
        }
    except:
        return {"params": "ERR", "quant": "ERR", "family": "ERR", "size_gb": 0}

def analyze_performance(model_name, response_json, duration_total):
    # Dati base
    load_dur = response_json.get("load_duration", 0) / 1e9
    eval_dur = response_json.get("eval_duration", 0) / 1e9
    gen_toks = response_json.get("eval_count", 0)
    tps = gen_toks / eval_dur if eval_dur > 0 else 0
    
    # Parsing
    raw = response_json["message"]["content"]
    json_valid = False
    narrative = raw
    
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r'(\{.*\})', clean, re.DOTALL)
        if match: 
            data = json.loads(match.group(1))
            json_valid = True
            narrative = data.get("descrizione_audio", raw)
    except: pass

    # --- CALCOLO METRICHE ---
    fidelity_score, missing = calculate_fidelity(USER_INPUT, narrative)
    tts_score, tts_errors = check_tts_safety(narrative)
    len_score, word_count = check_length_accuracy(narrative)
    ttr = calculate_ttr(narrative)
    
    # --- PUNTEGGIO FINALE PONDERATO ---
    final_score = 0
    
    # Somma dei punteggi standard
    final_score += (fidelity_score * WEIGHTS["fidelity"])
    final_score += (tts_score * WEIGHTS["tts_safety"]) 
    final_score += (len_score * WEIGHTS["length_accuracy"])
    final_score += (tps * WEIGHTS["speed"])
    final_score += (ttr * 100 * WEIGHTS["lexical_variety"]) 
    final_score += (load_dur * WEIGHTS["memory_load"]) 
    
    # APPLICAZIONE PENALITA BLOCCANTE PER JSON INVALIDO
    if not json_valid:
        final_score += PENALTY_BROKEN_JSON

    return {
        "json_valid": json_valid,
        "tps": round(tps, 2),
        "load_time": round(load_dur, 2),
        "fidelity_pct": fidelity_score,
        "missing_concepts": str(missing),
        "tts_score": tts_score,         
        "tts_errors": str(tts_errors),  
        "word_count": word_count,       
        "lexical_var_ttr": ttr,         
        "final_score": round(final_score, 2),
        "narrative_preview": narrative[:50].replace("\n", " ") + "..."
    }

# --- MAIN ---

def run_benchmark():
    run_id = get_next_run_id()

    print(f"Avvio Benchmark [Run ID: {run_id}] su {len(MODELS)} modelli.\n")
    results = []
    
    for model in MODELS:
        print(f"Testing: {model}...")
        specs = get_model_details(model)
        
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_INPUT}],
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 2048}
        }
        
        try:
            start = time.time()
            res = requests.post(f"{OLLAMA_API}/chat", json=payload)
            if res.status_code == 200:
                metrics = analyze_performance(model, res.json(), time.time() - start)
                
                row = {**{"Run_ID": run_id, "Model": model}, **specs, **metrics}
                
                results.append(row)
                
                # Feedback visivo sullo stato JSON
                json_status = "OK" if metrics['json_valid'] else "FAIL"
                print(f"   Score: {metrics['final_score']} | JSON: {json_status} | TTS Safe: {metrics['tts_score']}/100 | Fedelta: {metrics['fidelity_pct']}%")
            else:
                print(f"   HTTP Error {res.status_code}")
        except Exception as e:
            print(f"   Exception: {e}")

    if results:
        keys = [
            "Run_ID", "Model", "final_score", "tts_score", "tts_errors", 
            "fidelity_pct", "missing_concepts", "json_valid", 
            "tps", "load_time", "size_gb", "word_count", 
            "lexical_var_ttr", "params", "quant", "family", 
            "narrative_preview"
        ]
        
        file_exists = os.path.isfile(OUTPUT_FILE)
        
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(results)
            
        print(f"\nSalvataggio completato (Run ID: {run_id}) in: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()