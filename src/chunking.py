# src/chunking.py

def chunk_text_by_tokens(
    text,
    tokenizer,
    max_tokens,
    overlap,
):
    """Split a text string into a list of smaller string chunks based on token counts.

    This function encodes the input text into tokens, slices it into chunks of a
    maximum size while accounting for a specified token overlap, and decodes the
    tokens back into text strings. It contains a correction loop to ensure that
    the re-encoded string chunk strictly adheres to the max_tokens limit.

    Args:
        text (str): The input text to be chunked.
        tokenizer: The tokenizer object with `encode` and `decode` methods.
        max_tokens (int): The maximum allowable number of tokens per text chunk.
        overlap (int): The number of tokens that should overlap between
            consecutive chunks.

    Returns:
        list of str: A list of text chunks, each constrained by the max_tokens
        limit.
    """
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
