# src/retrieval.py

import numpy as np


def retrieve_relevant_chunks(
    query,
    store,
    k=10,
):
    """Retrieve the top K most similar text chunks from the vector store index.

    Args:
        query (str): The user query to search for.
        store: The vector store object containing the embedding model, index, and metadata.
        k (int, optional): The number of closest chunks to retrieve. Defaults to 10.

    Returns:
        list of dict: A list of dicts containing the chunk text, document source,
        chunk ID, and similarity score, ordered by relevance.
    """
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
