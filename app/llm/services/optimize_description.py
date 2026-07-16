from app.llm.factory import LLMFactory
import json
from app.schemas import DescriptionResponse
from app.utils.text_processing import smart_chunking

def generate_optimized_description(original_text: str, model_name: str = "llama3.1:8b") -> DescriptionResponse:
    """
    Riscrive una descrizione turistica ottimizzandola per l'ascolto (Storytelling Audio).

    Questa funzione utilizza un LLM per trasformare un testo grezzo o breve in una narrazione
    coinvolgente, stile "divulgatore culturale". Gestisce inoltre la suddivisione del testo
    in segmenti (chunks) per facilitare la sintesi vocale (TTS).

    Args:
        original_text (str): Il testo di input (es. "Statua del 1500 in marmo").
        model_name (str, optional): Il modello LLM da utilizzare (default: "llama3.1:8b").

    Returns:
        DescriptionResponse: Un oggetto contenente:
            - full_text_optimized (str): Il testo riscritto dall'AI.
            - tts_chunks (List[str]): Il testo diviso in frasi brevi per il TTS.
            - model_used (str): Il nome del modello utilizzato.

    Behavior:
        - **Persona:** Agisce come un esperto d'arte carismatico.
        - **Fact-Checking:** Include istruzioni esplicite per evitare allucinazioni su date/nomi.
        - **Fallback:** In caso di errore dell'LLM (timeout, crash), restituisce il testo originale
          senza ottimizzazioni per garantire la continuità del servizio.
    """
    
    # --- 1. PERSONA & STILE (Pattern Persona) ---
    # Definisce il tono di voce: fluido, elegante, non accademico ma autorevole.
    system_role = (
        "Sei un Divulgatore Culturale esperto e carismatico (stile Alberto Angela). "
        "Scrivi per un'applicazione di Realtà Aumentata che guida i turisti in Italia. "
        "Il tuo italiano è fluido, elegante ed evocativo. "
        "Il tuo obiettivo è emozionare l'utente, facendogli notare la bellezza di ciò che ha davanti."
    )

    # --- 2. CONTEXT & GUIDELINES (Constraint Pattern) ---
    # Vincoli specifici per evitare errori comuni (es. anacronismi) e migliorare l'immersione.
    guidelines = """
    LINEE GUIDA UNIVERSALI:
    1. NO ANACRONISMI: Fai estrema attenzione ai materiali e alle tecnologie. Non citare materiali moderni se l'opera è antica o rinascimentale.
    2. FOCUS SUL VISIBILE: Descrivi l'impatto visivo del monumento (le dimensioni, i materiali, i giochi di luce, i colori). Invita l'utente a guardare i dettagli.
    3. GRAMMATICA IMPECCABILE: Evita frasi passive, ripetizioni o traduzioni letterali. Usa un lessico ricco ma comprensibile.
    4. ESPANSIONE INTELLIGENTE: Usa il 'Testo di Base' come fonte di verità. Puoi arricchirlo con dettagli atmosferici o contesto storico generale, ma NON inventare date o nomi specifici se non sei sicuro al 100%.
    """

    # --- 3. FACT CHECKING (Safety Pattern) ---
    # Istruzioni critiche per ridurre le "allucinazioni" dell'AI.
    fact_checking_protocol = """
    PROTOCOLLO DI FACT-CHECKING E VERITÀ STORICA (PRIORITÀ ASSOLUTA):
    1. Poiché devi espandere il testo, userai le tue conoscenze interne. FAI ATTENZIONE.
    2. VERIFICA ogni data, nome di imperatore o evento storico che aggiungi. Devono essere REALI.
    3. NON INVENTARE MAI DETTAGLI. Se non sei sicurissimo di un anno specifico, usa termini più generici (es. "all'inizio del primo secolo") piuttosto che rischiare un numero sbagliato.
    4. Se il testo originale contiene un errore palese, correggilo basandoti sulla tua conoscenza enciclopedica.
    """

    # --- 4. OUTPUT FORMATTING (Format Pattern) ---
    # Pulisce il testo per il motore audio (niente markdown, niente elenchi puntati).
    constraints = """
    VINCOLI DI LUNGHEZZA E FORMATO (OBBLIGATORI):
    1. LUNGHEZZA: Devi generare un testo di ALMENO 130-150 parole. Se il testo originale è breve, USA LE TUE CONOSCENZE per arricchirlo con dettagli storici, curiosità e descrizioni visive pertinenti.
    2. AUDIO CLEANING: 
       - Scrivi i numeri in lettere se necessario per la fluidità.
       - Niente parentesi, caratteri speciali o elenchi puntati.
    3. Ogni frase deve avere senso compiuto.
    """

    # --- COSTRUZIONE PROMPT ---
    user_prompt = f"""
    {system_role}
    
    TESTO ORIGINALE: "{original_text}"
    
    Compito: Riscrivi il testo rendendolo accattivante e suddividilo per la sintesi vocale.
    {guidelines}
    {fact_checking_protocol}
    {constraints}
    
    Rispondi SOLO con il testo ottimizzato.
    """

    try:
        # Invocazione Factory LLM
        llm_engine = LLMFactory.get_engine(model_name)
        
        # Generazione Testo
        raw_content = llm_engine.generate(prompt=user_prompt, system_prompt=system_role)
        
        # Post-Processing: Chunking intelligente per TTS
        # Divide il testo lungo in segmenti logici (frasi) per evitare pause innaturali nell'audio.
        chunks = smart_chunking(raw_content)
        
        return DescriptionResponse(
            full_text_optimized=raw_content,
            tts_chunks=chunks,
            model_used=model_name,
            success=True,
            error_message=None
        )

    except Exception as e:
        error_str = str(e)
        print(f"Errore LLM Descrizione: {error_str}")

        # Mapping errori tecnici -> Messaggi leggibili (stesso pattern di optimize_markdown.py)
        friendly_error = "Errore generico AI durante la generazione della descrizione."
        if "Connection refused" in error_str or "No connection" in error_str:
            friendly_error = "Impossibile connettersi a Ollama. Assicurati che sia attivo."
        elif "model" in error_str and "not found" in error_str:
            friendly_error = "Modello LLM non trovato sul server."
    
        # --- FALLBACK STRATEGY ---
        # Se l'AI fallisce, il sistema deve rimanere operativo.
        # Restituiamo il testo originale così com'è.
        return DescriptionResponse(
            full_text_optimized=original_text,
            tts_chunks=[original_text], # Unico chunk
            model_used=f"Error ({model_name}) - Fallback to Original",
            success=False,
            error_message=f"{friendly_error} (Dettagli: {error_str})"
        )