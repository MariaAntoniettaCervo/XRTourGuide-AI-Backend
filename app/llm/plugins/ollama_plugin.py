import ollama
from app.interfaces.llm_interface import LLMInterface
from app.settings import global_settings

class OllamaLLM(LLMInterface):
    def generate(self, prompt: str, system_prompt: str = "Sei un assistente utile.") -> str:
        model = global_settings.llm_model_name
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ]
        )
        return response['message']['content']