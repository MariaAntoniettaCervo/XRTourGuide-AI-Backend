from app.llm.plugins.ollama_plugin import OllamaLLM

class LLMFactory:
    @staticmethod
    def get_engine():
        return OllamaLLM()