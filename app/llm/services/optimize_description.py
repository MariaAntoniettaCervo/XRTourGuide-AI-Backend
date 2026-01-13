from app.llm.factory import LLMFactory
import json
from app.schemas import DescriptionResponse
from app.utils.text_processing import smart_chunking

def generate_optimized_description(original_text: str, model_name: str = "qwen2.5:7b") -> DescriptionResponse:
    
    # --- PERSONA & STILE --- Pattern Persona ed Audience
    system_role = (
        "Sei un Divulgatore Culturale esperto e carismatico (stile Alberto Angela). "
        "Scrivi per un'applicazione di Realtà Aumentata che guida i turisti in Italia. "
        "Il tuo italiano è fluido, elegante ed evocativo. "
        "Il tuo obiettivo è emozionare l'utente, facendogli notare la bellezza di ciò che ha davanti."
    )

    # --- Context Pattern ---
    guidelines = """
    LINEE GUIDA UNIVERSALI:
    1. NO ANACRONISMI: Fai estrema attenzione ai materiali e alle tecnologie. Non citare materiali moderni se l'opera è antica o rinascimentale.
    2. FOCUS SUL VISIBILE: Descrivi l'impatto visivo del monumento (le dimensioni, i materiali, i giochi di luce, i colori). Invita l'utente a guardare i dettagli.
    3. GRAMMATICA IMPECCABILE: Evita frasi passive, ripetizioni o traduzioni letterali. Usa un lessico ricco ma comprensibile.
    4. ESPANSIONE INTELLIGENTE: Usa il 'Testo di Base' come fonte di verità. Puoi arricchirlo con dettagli atmosferici o contesto storico generale, ma NON inventare date o nomi specifici se non sei sicuro al 100%.
    """
    # ---  FACT CHECKING PATTERN ---
    fact_checking_protocol = """
    PROTOCOLLO DI FACT-CHECKING E VERITÀ STORICA (PRIORITÀ ASSOLUTA):
    1. Poiché devi espandere il testo, userai le tue conoscenze interne. FAI ATTENZIONE.
    2. VERIFICA ogni data, nome di imperatore o evento storico che aggiungi. Devono essere REALI.
    3. NON INVENTARE MAI DETTAGLI. Se non sei sicurissimo di un anno specifico, usa termini più generici (es. "all'inizio del primo secolo") piuttosto che rischiare un numero sbagliato.
    4. Se il testo originale contiene un errore palese, correggilo basandoti sulla tua conoscenza enciclopedica.
    """

    # --- Template Pattern ---
    constraints = """
    VINCOLI DI LUNGHEZZA E FORMATO (OBBLIGATORI):
    1. LUNGHEZZA: Devi generare un testo di ALMENO 130-150 parole. Se il testo originale è breve, USA LE TUE CONOSCENZE per arricchirlo con dettagli storici, curiosità e descrizioni visive pertinenti.
    2. AUDIO CLEANING: 
       - Scrivi i numeri in lettere se necessario per la fluidità.
       - Niente parentesi, caratteri speciali o elenchi puntati.
    3. Ogni frase deve avere senso compiuto.
    """


    # --- Costruzione Prompt ---
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
        llm_engine = LLMFactory.get_engine(model_name)
        
        raw_content = llm_engine.generate(prompt=user_prompt, system_prompt=system_role)
        
        chunks = smart_chunking(raw_content)
        
        return DescriptionResponse(
            full_text_optimized=raw_content,
            tts_chunks=chunks,
            model_used=model_name
        )

    except Exception as e:
        print(f"Errore LLM Descrizione: {e}")
        # Fallback sicuro
        return DescriptionResponse(
            full_text_optimized=original_text,
            tts_chunks=[original_text],
            model_used=f"Error ({model_name}) - Fallback to Original"
        )