import requests
import time
import json
import csv
import re
import os
import argparse
import psutil
import subprocess
import datetime

# --- CONFIGURAZIONE ---
MODELS = [
    "qwen2.5:7b",
    "llama3.1:8b",
    "mistral-nemo",
    "phi3.5"
]

OLLAMA_API = "http://localhost:11434/api"
BASELINE_VRAM = 0.0

# Pesi per il calcolo del punteggio finale
WEIGHTS = {
    "tts_safety": 1.5, "fidelity": 2.0, "speed": 1.5,
    "length_accuracy": 0.5, "lexical_variety": 0.5,
    "memory_load": -1.0, "ram_usage": -2.0
}

PENALTY_BROKEN_JSON = -1000.0

# --- PROMPT ---
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

# --- HARDWARE MODE (rilevato una volta all'avvio) ---

def detect_hardware_mode():
    """
    Rileva esplicitamente se è presente una GPU NVIDIA funzionante, invece di
    dedurlo a posteriori dal delta di VRAM osservato durante i run (che a
    volte risultava 0.0 su GPU per motivi estranei al benchmark).
    Ritorna 'gpu' o 'cpu'.
    """
    try:
        subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8", timeout=10)
        return "gpu"
    except Exception:
        return "cpu"


HARDWARE_MODE = detect_hardware_mode()
OUTPUT_FILE = f"benchmark_results_{HARDWARE_MODE}.csv"
FAILED_LOG_FILE = f"benchmark_failed_runs_{HARDWARE_MODE}.csv"
SUMMARY_FILE = f"benchmark_summary_{HARDWARE_MODE}.csv"

RESULT_FIELDS = [
    "Run_ID", "Timestamp", "Model", "HardwareMode", "final_score",
    "ram_delta_gb", "vram_delta_gb", "json_valid",
    "tts_score", "tts_errors", "fidelity_pct", "missing_concepts",
    "tps", "load_time", "size_gb", "word_count",
    "lexical_var_ttr", "params", "quant", "family",
    "narrative_preview"
]

FAILED_FIELDS = ["Run_ID", "Timestamp", "Model", "HardwareMode", "ErrorReason"]

# --- FUNZIONI UTILI ---

def get_system_ram_gb():
    try:
        mem = psutil.virtual_memory()
        return round(mem.used / (1024**3), 2)
    except:
        return 0.0

def get_vram_gb():
    """Ottiene l'uso della VRAM attuale tramite nvidia-smi. Se non disponibile, ritorna 0.0 (CPU-only)."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        total_mem_mb = sum(float(x) for x in result.strip().split('\n') if x.strip())
        return round(total_mem_mb / 1024, 2)
    except:
        return 0.0

def quick_unload(model_name):
    """Invia segnale di unload rapido."""
    try:
        requests.post(f"{OLLAMA_API}/chat", json={"model": model_name, "keep_alive": 0}, timeout=30)
    except:
        pass

def get_running_models():
    """Interroga l'API per vedere cosa è caricato in memoria."""
    try:
        res = requests.get(f"{OLLAMA_API}/ps", timeout=30)
        if res.status_code == 200:
            return res.json().get('models', [])
    except:
        pass
    return []

def list_installed_models():
    """Ritorna un set con i nomi dei modelli installati in Ollama."""
    try:
        res = requests.get(f"{OLLAMA_API}/tags", timeout=30)
        if res.status_code == 200:
            models = res.json().get("models", [])
            return {m.get("name") for m in models if m.get("name")}
    except:
        pass
    return set()

def ensure_model_installed(model_name):
    """
    Se il modello non è presente in locale, esegue il pull tramite API.
    Ritorna (True, None) se il modello è disponibile, (False, motivo) altrimenti.
    """
    installed = list_installed_models()
    if model_name in installed or (model_name + ":latest") in installed:
        return True, None

    print(f"   Modello non trovato localmente: {model_name}. Download (pull) in corso...")
    try:
        res = requests.post(
            f"{OLLAMA_API}/pull",
            json={"name": model_name, "stream": False},
            timeout=60 * 60  # download può richiedere tempo
        )
        if res.status_code == 200:
            print(f"   Download completato: {model_name}")
            return True, None
        else:
            reason = f"PULL_FAILED_HTTP_{res.status_code}: {res.text[:200]}"
            print(f"   Pull fallito ({res.status_code}): {res.text[:200]}")
            return False, reason
    except Exception as e:
        reason = f"PULL_EXCEPTION: {e}"
        print(f"   Errore durante pull di {model_name}: {e}")
        return False, reason

def initial_cleanup():
    """
    Esegue la pulizia profonda prima del benchmark.
    Identifica i modelli attivi, li scarica e attende la stabilizzazione.
    """
    print("--- FASE 1: PULIZIA INIZIALE SISTEMA ---")
    active_models = get_running_models()

    if active_models:
        print(f"Rilevati {len(active_models)} modelli attivi. Scaricamento forzato in corso...")
        for m in active_models:
            name = m.get('name', '')
            if name:
                print(f"   Scaricamento: {name}")
                quick_unload(name)

        print("   Attesa stabilizzazione VRAM (5s)...")
        time.sleep(5)
    else:
        print("   Nessun modello attivo rilevato. Sistema pulito.")

    vram = get_vram_gb()
    print(f"   VRAM Attuale post-pulizia: {vram} GB")
    return vram

def get_next_run_id():
    if not os.path.isfile(OUTPUT_FILE):
        return 1
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            ids = [int(row.get('Run_ID', 0)) for row in reader if row.get('Run_ID', '').isdigit()]
            if not ids:
                return 1
            return max(ids) + 1
    except:
        return 1

def log_failed_run(run_id, model, reason):
    """Registra un run fallito in modo esplicito, invece di ometterlo in silenzio."""
    file_exists = os.path.isfile(FAILED_LOG_FILE)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(FAILED_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FAILED_FIELDS)
            if not file_exists or os.path.getsize(FAILED_LOG_FILE) == 0:
                writer.writeheader()
            writer.writerow({
                "Run_ID": run_id,
                "Timestamp": timestamp,
                "Model": model,
                "HardwareMode": HARDWARE_MODE,
                "ErrorReason": reason
            })
    except Exception as e:
        print(f"   [WARN] Impossibile scrivere su {FAILED_LOG_FILE}: {e}")

# --- METRICHE DI ANALISI ---

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
    diff = (min_w - words) if words < min_w else (words - max_w)
    score = max(0, 100 - (diff * 2))
    return score, words

def calculate_ttr(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    if not words:
        return 0
    unique_words = set(words)
    return round(len(unique_words) / len(words), 2)

def calculate_fidelity(user_input, generated_text):
    clean_in = re.sub(r'[^\w\s]', '', user_input.lower())
    input_keys = {w for w in clean_in.split() if len(w) > 3 and w not in ["della", "sono", "questo"]}
    if not input_keys:
        return 100, []
    gen_lower = generated_text.lower()
    matches = 0
    missing = []
    for k in input_keys:
        if k[:-1] in gen_lower:
            matches += 1
        else:
            missing.append(k)
    score = (matches / len(input_keys)) * 100
    return round(score, 1), missing

def get_model_details(model_name):
    try:
        res = requests.post(f"{OLLAMA_API}/show", json={"name": model_name}, timeout=60).json()
        details = res.get("details", {})
        size_gb = 0.0
        try:
            tags = requests.get(f"{OLLAMA_API}/tags", timeout=60).json()
            for m in tags.get('models', []):
                if m.get('name') == model_name or m.get('name') == model_name + ":latest":
                    size_gb = round(m.get('size', 0) / (1024**3), 2)
                    break
        except:
            pass
        return {
            "params": details.get("parameter_size", "N/A"),
            "quant": details.get("quantization_level", "N/A"),
            "family": details.get("family", "N/A"),
            "size_gb": size_gb
        }
    except:
        return {"params": "ERR", "quant": "ERR", "family": "ERR", "size_gb": 0.0}

def analyze_performance(model_name, response_json, duration_total, ram_delta, vram_load_absolute):
    load_dur = response_json.get("load_duration", 0) / 1e9
    eval_dur = response_json.get("eval_duration", 0) / 1e9
    gen_toks = response_json.get("eval_count", 0)
    tps = gen_toks / eval_dur if eval_dur > 0 else 0

    raw = response_json.get("message", {}).get("content", "")
    json_valid = False
    narrative = raw

    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r'(\{.*\})', clean, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            json_valid = True
            narrative = data.get("descrizione_audio", raw)
    except:
        pass

    fidelity_score, missing = calculate_fidelity(USER_INPUT, narrative)
    tts_score, tts_errors = check_tts_safety(narrative)
    len_score, word_count = check_length_accuracy(narrative)
    ttr = calculate_ttr(narrative)

    # --- PUNTEGGIO FINALE ---
    final_score = 0.0
    final_score += (fidelity_score * WEIGHTS["fidelity"])
    final_score += (tts_score * WEIGHTS["tts_safety"])
    final_score += (len_score * WEIGHTS["length_accuracy"])
    final_score += (tps * WEIGHTS["speed"])
    final_score += (ttr * 100 * WEIGHTS["lexical_variety"])
    final_score += (load_dur * WEIGHTS["memory_load"])
    final_score += (ram_delta * WEIGHTS["ram_usage"])

    if not json_valid:
        final_score += PENALTY_BROKEN_JSON

    return {
        "json_valid": json_valid,
        "tps": round(tps, 2),
        "load_time": round(load_dur, 2),
        "ram_delta_gb": ram_delta,
        "vram_delta_gb": vram_load_absolute,
        "fidelity_pct": fidelity_score,
        "missing_concepts": str(missing),
        "tts_score": tts_score,
        "tts_errors": str(tts_errors),
        "word_count": word_count,
        "lexical_var_ttr": ttr,
        "final_score": round(final_score, 2),
        "narrative_preview": narrative[:50].replace("\n", " ") + "..."
    }

# --- REPORT DI SINTESI (json_valid_rate isolato dal final_score) ---

def generate_summary_report():
    """
    Legge il CSV dei risultati della condizione hardware corrente e produce
    un riepilogo per modello: run totali, run json-validi, json_valid_rate
    e final_score medio (calcolato solo sui run json-validi, per non farlo
    inquinare dalla penalità PENALTY_BROKEN_JSON).
    """
    if not os.path.isfile(OUTPUT_FILE):
        print(f"\n[SUMMARY] Nessun file {OUTPUT_FILE} trovato, report non generato.")
        return

    per_model = {}
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("Run_ID", "")
            if not run_id.isdigit():
                continue  # salta righe separatore ("------") o vuote
            model = row.get("Model", "UNKNOWN")
            entry = per_model.setdefault(model, {"total": 0, "valid": 0, "scores_valid": []})
            entry["total"] += 1
            is_valid = str(row.get("json_valid", "")).strip().lower() == "true"
            if is_valid:
                entry["valid"] += 1
                try:
                    entry["scores_valid"].append(float(row.get("final_score", 0)))
                except ValueError:
                    pass

    if not per_model:
        print(f"\n[SUMMARY] Nessun run valido trovato in {OUTPUT_FILE}.")
        return

    summary_rows = []
    print(f"\n--- SUMMARY REPORT ({HARDWARE_MODE.upper()}) ---")
    for model, stats in per_model.items():
        total = stats["total"]
        valid = stats["valid"]
        rate = round(valid / total, 3) if total else 0.0
        avg_score_valid = round(sum(stats["scores_valid"]) / len(stats["scores_valid"]), 2) if stats["scores_valid"] else "N/A"
        print(f"   {model:15s} | run: {total:3d} | json_valid: {valid:3d} | json_valid_rate: {rate:.1%} | avg_final_score (solo validi): {avg_score_valid}")
        summary_rows.append({
            "Model": model,
            "HardwareMode": HARDWARE_MODE,
            "total_runs": total,
            "json_valid_runs": valid,
            "json_valid_rate": rate,
            "avg_final_score_valid_only": avg_score_valid
        })

    with open(SUMMARY_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Model", "HardwareMode", "total_runs", "json_valid_runs",
            "json_valid_rate", "avg_final_score_valid_only"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"   Report salvato in: {SUMMARY_FILE}")

# --- SINGOLA RUN ---

def run_single_benchmark():
    """Esegue un'unica sessione di benchmark (tutti i modelli in MODELS) e la appende al CSV."""
    global BASELINE_VRAM
    run_id = get_next_run_id()
    print(f"\nAvvio Benchmark [Run ID: {run_id}] [Hardware: {HARDWARE_MODE.upper()}]\n")

    # 1. CLEANUP e CALCOLO TARA
    BASELINE_VRAM = initial_cleanup()

    print(f"\n--- BASELINE VRAM DEFINITIVA: {BASELINE_VRAM} GB ---")
    if BASELINE_VRAM > 2.0:
        print("ATTENZIONE: La VRAM di base e' ancora alta (>2GB).")
    else:
        print("VRAM Ottimale per il test.\n")

    results = []

    for model in MODELS:
        print(f"Testing: {model}...")

        # 0) Assicura che il modello sia installato (evita HTTP 404)
        installed, install_reason = ensure_model_installed(model)
        if not installed:
            print(f"   Skip: impossibile installare {model}")
            log_failed_run(run_id, model, install_reason or "INSTALL_FAILED_UNKNOWN")
            print("-" * 30)
            continue

        # Unload rapido precauzionale
        quick_unload(model)

        specs = get_model_details(model)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_INPUT}
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 2048}
        }

        try:
            ram_before = get_system_ram_gb()

            start = time.time()
            res = requests.post(f"{OLLAMA_API}/chat", json=payload, timeout=60 * 10)

            ram_after = get_system_ram_gb()
            vram_end = get_vram_gb()

            # Calcoli Delta
            ram_delta = max(0.0, round(ram_after - ram_before, 2))

            # Carico VRAM Assoluto (Totale attuale - Tara iniziale pulita)
            vram_load_calc = max(0.0, round(vram_end - BASELINE_VRAM, 2))

            if res.status_code == 200:
                metrics = analyze_performance(model, res.json(), time.time() - start, ram_delta, vram_load_calc)
                row = {
                    "Run_ID": run_id, "Model": model, "HardwareMode": HARDWARE_MODE,
                    **specs, **metrics
                }
                results.append(row)

                json_status = "OK" if metrics['json_valid'] else "FAIL"

                print(f"   Score: {metrics['final_score']} | RAM Delta: +{metrics['ram_delta_gb']}GB | JSON: {json_status} | TTS Safe: {metrics['tts_score']}/100 | Fedelta: {metrics['fidelity_pct']}%")
                print(f"   (VRAM Load: {metrics['vram_delta_gb']}GB [Tot: {vram_end} - Base: {BASELINE_VRAM}])")

                # Messaggio RAM più corretto per GPU vs CPU-only
                if metrics['ram_delta_gb'] > 0.5:
                    if HARDWARE_MODE == "gpu":
                        print(f"   RAM Overhead: +{metrics['ram_delta_gb']} GB (parte del carico resta su CPU)")
                    else:
                        print(f"   CPU-only RAM Load: +{metrics['ram_delta_gb']} GB")

                # Unload rapido post-test
                quick_unload(model)
                print("-" * 30)

            else:
                reason = f"HTTP_{res.status_code}: {res.text[:200]}"
                print(f"   HTTP Error {res.status_code} | {res.text[:200]}")
                log_failed_run(run_id, model, reason)
                print("-" * 30)

        except Exception as e:
            print(f"   Exception: {e}")
            log_failed_run(run_id, model, f"EXCEPTION: {e}")
            print("-" * 30)

    if results:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in results:
            row["Timestamp"] = current_time

        file_exists = os.path.isfile(OUTPUT_FILE)

        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            if file_exists and os.path.getsize(OUTPUT_FILE) > 0:
                f.write('\n\n')

            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction='ignore')

            if not file_exists:
                writer.writeheader()

            writer.writerow({k: "------" for k in RESULT_FIELDS})
            writer.writerows(results)

        print(f"\nSalvataggio completato (Run ID: {run_id}) in: {OUTPUT_FILE}")
    else:
        print(f"\nNessun risultato salvato per il Run ID {run_id}: nessun modello testato con successo.")

    return len(results) > 0

# --- MAIN ---

def run_benchmark(num_runs=1):
    print(f"=== Sessione di benchmark: {num_runs} run x {len(MODELS)} modelli | Hardware: {HARDWARE_MODE.upper()} ===")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Failed log: {FAILED_LOG_FILE}")

    for i in range(1, num_runs + 1):
        print(f"\n########## SESSIONE {i}/{num_runs} ##########")
        run_single_benchmark()
        if i < num_runs:
            print("Pausa di raffreddamento (10s) prima della prossima run...")
            time.sleep(10)

    generate_summary_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark modelli LLM per XRTourGuide-AI-Backend")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Numero di run consecutive da eseguire in questa sessione (proposta: 8-10 per raggiungere un dataset statisticamente robusto per condizione)"
    )
    args = parser.parse_args()
    run_benchmark(num_runs=args.runs)