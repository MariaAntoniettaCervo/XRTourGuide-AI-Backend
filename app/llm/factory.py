from app.llm.plugins.ollama_plugin import OllamaLLM

class LLMFactory:
    @staticmethod
    def get_engine(model_name: str): 
        return OllamaLLM(model=model_name)