# src/chunking.py

def chunk_text_by_tokens(
    text,
    tokenizer,
    max_tokens,
    overlap,
):
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    chunks = []

    start = 0
    n = len(tokens)

    while start < n:

        end = min(start + max_tokens, n)

        chunk_tokens = tokens[start:end]

        chunk = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
        )

        encoded_len = len(
            tokenizer.encode(
                chunk,
                add_special_tokens=False,
            )
        )

        while encoded_len > max_tokens:

            chunk_tokens = chunk_tokens[:-1]

            chunk = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
            )

            encoded_len = len(
                tokenizer.encode(
                    chunk,
                    add_special_tokens=False,
                )
            )

        chunks.append(chunk)

        step = max_tokens - overlap

        if step <= 0:
            step = 1

        start += step

    return chunks
