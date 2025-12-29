from app.llm.factory import LLMFactory
from app.schemas import MarkdownFixResponse
import logging

# Scegli il modello
MODEL_NAME = "qwen2.5:7b" 

def fix_text_logic(text: str, tone: str) -> str:
    """
    Invia il testo a Ollama per la correzione ortografica, grammaticale 
    e di formattazione Markdown.
    """
    print(f"FIXING MARKDOWN (Tone: {tone})...")

    system_prompt = (
        "Sei un editor esperto e un correttore di bozze professionale. "
        "Il tuo compito è riscrivere il testo fornito dall'utente rispettando queste regole:\n"
        "1. Correggi errori grammaticali e di sintassi.\n"
        "2. Migliora la fluidità del testo mantenendo il significato originale.\n"
        "3. Mantieni o migliora la formattazione MARKDOWN (grassetto, elenchi, titoli).\n"
        "4. Non aggiungere commenti, introduzioni o saluti. Restituisci SOLO il testo corretto.\n"
        f"5. Adotta un tono: {tone}."
    )

    try:
        # 2. Ottieni il motore attivo (Astrazione)
        llm_engine = LLMFactory.get_engine()
        
        # 3. Genera il testo usando l'interfaccia generica
        # Non ci interessa se dietro c'è Ollama, GPT-4 o un Mock
        fixed_content = llm_engine.generate(prompt=text, system_prompt=system_prompt)
        
       
        return MarkdownFixResponse(
            original_text=text,
            fixed_text=fixed_content,
            success=True,
            error_message=None
        )

    except Exception as e:
        error_str = str(e)
        print(f"Errore LLM Fix: {error_str}")
        
        # Capiamo che tipo di errore è
        friendly_error = "Errore generico AI."
        if "Connection refused" in error_str or "No connection" in error_str:
            friendly_error = "Impossibile connettersi a Ollama. Assicurati che sia attivo."
        elif "model" in error_str and "not found" in error_str:
            friendly_error = f"Modello '{MODEL_NAME}' non trovato."

        # CASO ERRORE (Restituiamo comunque il testo originale per non interrompere il flusso)
        return MarkdownFixResponse(
            original_text=text,
            fixed_text=text, # Fallback: restituiamo il testo originale 
            success=False,
            error_message=f"{friendly_error} (Dettagli: {error_str})"
        )