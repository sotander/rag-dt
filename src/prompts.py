# src/prompts.py

SYSTEM_PROMPT = """
You are a precise corporate communications assistant.

Rules:
- Use ONLY information found in the provided context.
- Cite source document names when possible.
- Do not invent facts.
- When mentioning a title, performer, organization, venue, or named entity, try to fix the capitalization if needed.
- If the answer is not present in the context, respond exactly:

I cannot find the answer in the provided documents.
"""


def build_messages(
    query,
    retrieved_chunks,
):
    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Source: {chunk['document']}]\n{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion:\n\n{query}",
        },
    ]
