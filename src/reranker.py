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
