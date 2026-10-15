
class ByteTokenizer:
    """Represent text as UTF-8 bytes, with one token per byte."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        # Generated sequences may contain incomplete or invalid UTF-8.
        return bytes(token_ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = ByteTokenizer()

    text = "The robot moved."

    ids = tokenizer.encode(text)

    print("Text:", text)
    print("Token IDs:", ids)
    print("Vocabulary size:", tokenizer.vocab_size)

    for token_id in ids:
        print(token_id, repr(tokenizer.decode([token_id])))

    assert tokenizer.decode(ids) == text
