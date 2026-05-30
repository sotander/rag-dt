# src/retrieval.py

import numpy as np


def retrieve_relevant_chunks(
    query,
    store,
    k=10,
):
    query_embedding = store.embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = store.index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx == -1:
            continue

        results.append(
            {
                "text": store.chunks[idx],
                "document": store.metadata[idx]["document"],
                "chunk_id": store.metadata[idx]["chunk_id"],
                "score": float(score),
            }
        )

    return results
