# src/rag.py

import torch

from .retrieval import retrieve_relevant_chunks
from .prompts import build_messages
from .reranker import rerank_chunks


def fit_context_into_budget(
    query,
    chunks,
    tokenizer,
    max_input_tokens,
):
    selected = []

    for chunk in chunks:
        candidate = selected + [chunk]

        messages = build_messages(
            query,
            candidate,
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        token_count = len(
            tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
        )

        if token_count > max_input_tokens:
            break

        selected.append(chunk)

    return selected


def ask(
    query,
    store,
    model,
    tokenizer,
    reranker,
    k=10,
    max_new_tokens=256,
    max_input_tokens=6000,
    top_k=20,
    rerank_k=5,
):
    chunks = retrieve_relevant_chunks(
        query,
        store,
        k=top_k,
    )
    
    chunks = rerank_chunks(
        query,
        chunks,
        reranker,
        top_n=rerank_k,
    )

    chunks = fit_context_into_budget(
        query,
        chunks,
        tokenizer,
        max_input_tokens,
    )

    messages = build_messages(
        query,
        chunks,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[
        0,
        inputs.input_ids.shape[1]:,
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()
