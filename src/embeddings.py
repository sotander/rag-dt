# src/embeddings.py

from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name,
    device,
):
    model = SentenceTransformer(
        model_name,
        device=device,
    )

    tokenizer = model.tokenizer

    max_tokens = model.get_max_seq_length() - 2

    return model, tokenizer, max_tokens
