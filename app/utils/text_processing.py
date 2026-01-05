import re

def smart_chunking(text: str, max_chars: int = 200) -> list[str]:
    # Pulisce spazi doppi
    text = re.sub(r'\s+', ' ', text).strip()
    # Taglia mantenendo i delimitatori
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\.\.\.)\s+', text)
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if not sentence.strip(): continue
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks