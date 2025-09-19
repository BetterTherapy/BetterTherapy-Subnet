import tiktoken

def count_words(text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    words = text.split()
    return len(words)

def count_gpt_4_tokens(text:str) -> int:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    return len(tokens)