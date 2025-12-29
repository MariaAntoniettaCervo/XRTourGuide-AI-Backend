from abc import ABC, abstractmethod

class TTSInterface(ABC):
    @abstractmethod
    def generate_audio(self, text: str, output_filename: str) -> bool:
        pass