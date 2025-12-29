from app.llm.plugins.ollama_plugin import OllamaLLM

class LLMFactory:
    @staticmethod
    def get_engine():
        # Per ora supportiamo solo Ollama, ma in futuro qui aggiungerai OpenAI/Claude
        return OllamaLLM()