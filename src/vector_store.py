# src/vector_store.py

import os
import faiss
import numpy as np

from .chunking import chunk_text_by_tokens


class FaissStore:
    def __init__(
        self,
        embedding_model,
        tokenizer,
        max_tokens,
        overlap,
    ):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.overlap = overlap

        self.chunks = []
        self.metadata = []

        dim = embedding_model.get_embedding_dimension()

        self.index = faiss.IndexFlatIP(dim)

    def ingest_document(
        self,
        file_path,
        filename,
    ):
        """Read, chunk, embed, and index a single text file into the FAISS store.
        Args:
            file_path (str): Path to the target text file.
            filename (str): Name of the file to store in the chunk metadata.
        """
        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            text = f.read()

        chunks = chunk_text_by_tokens(
            text,
            self.tokenizer,
            self.max_tokens,
            self.overlap,
        )

        embeddings = self.embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.index.add(
            embeddings.astype(np.float32)
        )

        for i, chunk in enumerate(chunks):

            self.chunks.append(chunk)

            self.metadata.append(
                {
                    "document": filename,
                    "chunk_id": i,
                }
            )

    def ingest_directory(
        self,
        data_dir,
    ):
        for filename in os.listdir(data_dir):

            if not filename.endswith(".txt"):
                continue

            self.ingest_document(
                os.path.join(data_dir, filename),
                filename,
            )
