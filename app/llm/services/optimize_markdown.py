from app.llm.factory import LLMFactory
from app.schemas import MarkdownFixResponse
import logging

def fix_markdown(text: str, tone: str, model_name: str = "qwen2.5:7b") -> str:
 
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
        llm_engine = LLMFactory.get_engine(model_name)
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
        
        friendly_error = "Errore generico AI."
        if "Connection refused" in error_str or "No connection" in error_str:
            friendly_error = "Impossibile connettersi a Ollama. Assicurati che sia attivo."
        elif "model" in error_str and "not found" in error_str:
            friendly_error = "Modello non trovato."

        # CASO ERRORE (Restituiamo comunque il testo originale per non interrompere il flusso)
        return MarkdownFixResponse(
            original_text=text,
            fixed_text=text, # Fallback: restituiamo il testo originale 
            success=False,
            error_message=f"{friendly_error} (Dettagli: {error_str})"
        )