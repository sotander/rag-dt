from sentence_transformers.cross_encoder import CrossEncoder

def load_reranker(
    model_name,
    device,
):
    return CrossEncoder(
        model_name,
        device=device,
    )


def rerank_chunks(
    query,
    retrieved_chunks,
    reranker,
    top_n=5,
):
    """Re-rank retrieved text chunks using a cross-encoder and return the top N.

    Args:
        query (str): The user query to score against the chunks.
        retrieved_chunks (list of dict): Dicts containing a "text" key.
        reranker: The re-ranking model with a `predict` method.
        top_n (int, optional): Number of top-scoring chunks to return. Defaults to 5.

    Returns:
        list of dict: The top_n highest-scoring chunks, sorted descending.
    """
    pairs = [
        (query, chunk["text"])
        for chunk in retrieved_chunks
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    for chunk, score in zip(
        retrieved_chunks,
        scores,
    ):
        chunk["rerank_score"] = float(score)

    reranked = sorted(
        retrieved_chunks,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_n]
