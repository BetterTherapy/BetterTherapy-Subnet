import tiktoken

def count_words(text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    words = text.split()
    return len(words)

def count_and_clip_tokens(text:str, max_tokens:int) -> tuple[int, str]:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return len(tokens), text
        
    clipped_tokens = tokens[:max_tokens]
    clipped_text = encoding.decode(clipped_tokens)
    return max_tokens, clipped_text