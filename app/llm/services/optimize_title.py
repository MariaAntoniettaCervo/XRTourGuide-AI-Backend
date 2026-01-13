import json
from app.schemas import TitleResponse
from app.llm.factory import LLMFactory  

def generate_optimized_title(original_title: str, model_name: str = "qwen2.5:7b") -> TitleResponse:
    # 1. PERSONA & STILE
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

    # 3. USER PROMPT
    user_prompt = f"""
    ANALISI CONTESTO: Il titolo attuale '{original_title}' è troppo generico o poco attraente.
    OBIETTIVO: Generare varianti che spingano l'utente a cliccare per saperne di più.
    
    TASK: Riscrivi il titolo '{original_title}'.
    
    {json_template}
    
    Rispondi SOLO col JSON, nessun altro commento.
    """

    print(f"Ottimizzazione titolo '{original_title}' in corso...")

    try:
        llm_engine = LLMFactory.get_engine(model_name)
        
        raw_content = llm_engine.generate(prompt=user_prompt, system_prompt=system_role)
        
        clean_json = raw_content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        raw_options = data.get("options", [original_title])
        sanitized_options = []
        
        for item in raw_options:
            if isinstance(item, str):
                sanitized_options.append(item)
            elif isinstance(item, dict):
                # Se è un dizionario (es. {'text': 'Titolo'}), prendiamo il primo valore
                val = list(item.values())[0] if item else original_title
                sanitized_options.append(str(val))
            else:
                sanitized_options.append(str(item))

        return TitleResponse(
            original=original_title,
            options=sanitized_options,
            best_option=data.get("best_option", original_title),
            model_used=model_name
        )

    except Exception as e:
        print(f"Errore generazione titoli: {e}")
        # Fallback sicuro: restituisci il titolo originale come unica opzione
        return TitleResponse(
            original=original_title,
            options=[original_title], 
            best_option=original_title,
            model_used=f"Error ({model_name}) - Fallback"
        )