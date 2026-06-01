from src.config import (
    DEVICE,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    OVERLAP,
    DATA_DIR,
    MAX_INPUT_TOKENS,
    GOOGLE_DRIVE_FILE_ID,
)

from src.embeddings import load_embedding_model
from src.vector_store import FaissStore
from src.llm import load_gemma_4bit
from src.rag import ask
from src.reranker import load_reranker
from src.data_loader import ensure_dataset


def main():
    ensure_dataset(
        GOOGLE_DRIVE_FILE_ID,
        DATA_DIR,
    )

    print("Loading embedding model...")
    embedding_model, tokenizer, max_tokens = load_embedding_model(
        EMBEDDING_MODEL_NAME,
        DEVICE,
    )

    print("Building vector store...")
    store = FaissStore(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=OVERLAP,
    )

    store.ingest_directory(DATA_DIR)

    print(f"Indexed {store.index.ntotal} chunks")

    print("Loading reranker...")
    reranker = load_reranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        DEVICE,
    )

    print("Loading LLM...")
    llm_model, llm_tokenizer = load_gemma_4bit(
        LLM_MODEL_NAME,
    )

    print("\nRAG system ready.")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("> ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        answer = ask(
            query=query,
            store=store,
            model=llm_model,
            tokenizer=llm_tokenizer,
            reranker=reranker,
            max_input_tokens=MAX_INPUT_TOKENS,
        )

        print()
        print(answer)
        print()


if __name__ == "__main__":
    main()
