import json
from app.schemas import TitleResponse
from app.llm.factory import LLMFactory  

def generate_optimized_title(original_title: str) -> TitleResponse:
    
    # 1. SYSTEM PROMPT: Definizione Persona e Audience
    # È meglio tenere qui le regole generali di comportamento
    system_role = (
        "Sei uno storico dell'arte specializzato in turismo ed in storytelling. "
        "Il tuo stile è conciso, evocativo e moderno. "
        "Il tuo pubblico sono turisti curiosi che usano lo smartphone per visualizzare i contenuti in AR. "
        "I titoli devono catturarli in meno di 1 secondo. "
        "Evita linguaggio troppo accademico o troppo formale."
    )

    # 2. TEMPLATE JSON (Istruzioni di formato)
    json_template = """
    FORMATO OUTPUT RICHIESTO (JSON PURO):
    Devi restituire SOLO un oggetto JSON con questa struttura esatta:
    {
        "options": [
            "Titolo Corto & Punchy (Max 25 car)",
            "Titolo Evocativo/Misterioso",
            "Titolo Domanda/Ingaggio"
        ],
        "best_option": "La migliore delle tre"
    }
    """

    # 3. USER PROMPT: Il Task specifico (COSA fare ora)
    user_prompt = f"""
    ANALISI CONTESTO: Il titolo attuale '{original_title}' è troppo generico o poco attraente.
    OBIETTIVO: Generare varianti che spingano l'utente a cliccare per saperne di più.
    
    TASK: Riscrivi il titolo '{original_title}'.
    
    {json_template}
    
    Rispondi SOLO col JSON, nessun altro commento.
    """

    print(f"Ottimizzazione titolo '{original_title}' in corso...")

    try:
        # 4. CHIAMATA ASTRATTA
        # Otteniamo il motore configurato (Ollama, OpenAI, ecc.)
        llm_engine = LLMFactory.get_engine()
        
        # Generiamo la risposta
        raw_content = llm_engine.generate(prompt=user_prompt, system_prompt=system_role)
        
        # 5. PARSING E PULIZIA
        # Rimuoviamo eventuali backtick del markdown (spesso i modelli aggiungono ```json)
        clean_json = raw_content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        return TitleResponse(
            original=original_title,
            options=data.get("options", [original_title]),
            best_option=data.get("best_option", original_title)
        )

    except Exception as e:
        print(f"Errore generazione titoli: {e}")
        # Fallback sicuro: restituisci il titolo originale come unica opzione
        return TitleResponse(
            original=original_title,
            options=[original_title], 
            best_option=original_title
        )