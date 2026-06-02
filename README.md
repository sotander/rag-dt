# RAG-DT: Retrieval-Augmented Generation for Deutsche Telekom Press Releases

## 1. Objective
This project implements a Retrieval-Augmented Generation (RAG) system for question answering over a collection of Deutsche Telekom press releases. The system retrieves relevant passages from the document collection and uses them as evidence for grounded answer generation. An LLM is instructed to answer exclusively from the retrieved context, provide source attribution, and indicate when the answer cannot be found in the available documents.

The solution **is deployed** in Google Colab with a T4 GPU. Clicking "Run All" will load the data, model weights, produce a test output and offer a clickable UI built on `ipwidgets`. Allow a few minutes for the process to finish, especially the model weight loading. Access is through: [this link](https://colab.research.google.com/drive/1MwULbEIZMAoLYzi69a035WB6SyVNOtqL?usp=sharing). Note that you need to be logged in to a Google account; other that that, the notebok doesn't require any API keys and while it might ask for the HF token, it doesn't require it.

If you want to deploy this on your own HW, use the `main.py` script but you will have to figure out your environment for your HW. While `requirements.txt` contains the dependencies, the CUDA version depends on your particular GPU.

## 2. Architecture
The system follows a two-stage retrieval architecture consisting of dense retrieval and neural reranking.

```text
                ┌──────────────────┐
                │  User Question   │
                └────────┬─────────┘
                         │
                         ▼
                ┌─────────────────────────┐
                │ BGE Small v1.5 Encoder  │
                └────────┬────────────────┘
                         │
                         ▼
                ┌─────────────────────────────────┐
                │      FAISS Index                │
                │   Top_k Candidates (10 default) │
                └────────┬────────────────────────┘
                         │
                         ▼
                ┌───────────────────────────────┐
                │ MiniLM Cross Encoder          │
                │     Top_k 5 Chunks (3 default)|
                └────────┬──────────────────────┘
                         │
                         ▼
                ┌─────────────────────────┐
                │ Gemma 4 E4B (4-bit)     │
                └────────┬────────────────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Answer  │
                    └──────────┘
```

Documents are split into overlapping token-based chunks. Each chunk is embedded and stored in a FAISS vector index. At query time, the user question is embedded using the same model and the index retrieves the most similar chunks. The retrieved candidates are reranked by a cross-encoder, which evaluates each query-chunk pair directly and produces a relevance score. The highest-ranked chunks are assembled into a context that is passed to the language model.

## 3. Models and Hyperparameters

Dense retrieval uses the `BAAI/bge-small-en-v1.5` embedding model. The model was selected because it provides strong retrieval quality while remaining small enough for rapid indexing and query execution. More powerful embedding models are available, but the expected gains are relatively small compared to the additional computational cost.

Retrieved chunks are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Dense retrieval is optimized for recall and often returns several semantically related passages. The reranker improves precision by evaluating each candidate together with the user query.

Answer generation is performed by `google/gemma-4-E4B-it` loaded in 4-bit quantized mode through BitsAndBytes. The model provides a reasonable balance between instruction-following ability, memory requirements, and inference speed. The 4-bit configuration allows the entire system to run on a standard T4 GPU in Colab or any GPU with over 12 GB of VRAM.

### 3.1 Note on Local LLMs vs Commercial LLMs accisible via APIs
I was running into issues with rate limitting with free-ties commercial LLMs through API. Therefore, I decided to run everything local. A good commercial LLM will likely be better in information synthesis and more complex queries in general.


The current configuration uses the following retrieval parameters:

| Parameter | Value |
|------------|--------|
| Chunk overlap | 64 tokens |
| Retrieval depth | 10 chunks (default) |
| Reranked output | 3 chunks (default) |
| Maximum prompt budget | 6000 tokens |
| Generation length | 256 tokens |

The overlap value was chosen to reduce information loss at chunk boundaries while avoiding excessive duplication. Retrieval depth is intentionally larger than the final context size because the reranker requires a sufficiently large candidate pool to improve ranking quality.

## 4. Deployment and UI

The project can be executed either locally or in Google Colab. **I STRONGLY suggest using the Colab** because local deployment will require creating a local env matching the used libraries and your HW (use `requirements.txt`).

The repository contains all retrieval, reranking, and generation code. During startup the system checks whether the document collection is already available. If the `data` directory does not exist, the documents are downloaded from Google Drive and extracted automatically.

The primary target environment is Google Colab. The selected models fit within the memory constraints of the free Colab GPU tier when Gemma is loaded in 4-bit mode. Typical GPU memory consumption is approximately 10–12 GB after loading the embedding model, reranker, and language model simultaneously. The system therefore remains deployable without specialized hardware.

The UI is a simple textbox implemented with ipwidgets in the Colab notebook.

## 5. Evaluation

Evaluation focused on factual retrieval, multi-document reasoning, and recognition of technical terminology. The retrieval component performed consistently well throughout the test suite. All evaluation questions were answered using the correct supporting documents, yielding an effective retrieval accuracy of 100% on the tested examples.

The system demonstrated the ability to combine information originating from multiple documents. For example, questions involving historical achievements and future deployment plans were answered using evidence retrieved from separate press releases. Technical telecommunications concepts such as NarrowBand IoT (NB-IoT) and the Security Edge Protection Proxy (SEPP) protocol were retrieved and interpreted correctly.

The **most significant error** occurred during answer generation rather than retrieval. In one case involving Power Usage Effectiveness (PUE), the language model incorrectly interpreted a statement describing an ideal efficiency value as an observed measurement and consequently reported a contradiction between documents that did not exist. The retrieved evidence itself was correct, indicating that the limitation lies in the model's interpretation of numerical information rather than in the retrieval pipeline.

Minor inconsistencies in citation formatting were also observed. While these do not affect answer correctness, they could complicate integration with downstream systems that rely on deterministic citation extraction.

The evaluation therefore suggests a clear distinction between retrieval quality and generation quality. Retrieval was consistently reliable on the evaluated questions, while remaining errors originated primarily from language-model reasoning and interpretation.


## 6. Performance

Performance measurements were collected on a Google Colab T4 GPU environment using the final retrieval, reranking, and generation pipeline.

| Stage | Time |
|---------|---------:|
| Retrieval | 16.2 ms |
| Reranking | 93.6 ms |
| Generation | 22.97 s |
| End-to-end | 23.40 s |

The measurements show that retrieval and reranking contribute only a small fraction of the total response time. More than 99% of the latency originates from language model inference. The retrieval pipeline is therefore unlikely to be the primary bottleneck for this document collection. Any significant reduction in response time would require a smaller generation model, more aggressive quantization, or more powerful inference hardware.


## 7. Limitations and Future Work

The assignment does not define detailed evaluation criteria, target user groups, latency requirements, or operational constraints. Consequently, the current evaluation demonstrates that the system functions correctly as a complete RAG pipeline, but it does not establish suitability for a specific production deployment.

If the system were to be used in practice, further development should be driven by the actual tasks users perform rather than by a small general-purpose test set. Future evaluation should focus on domain-specific query distributions, citation quality, hallucination rates, numerical reasoning, response latency, and user satisfaction. The resulting requirements would then guide decisions regarding model selection, retrieval strategy, context construction, and infrastructure.

## 8. License

All major components of the retrieval pipeline are distributed under permissive open-source licenses (MIT or Apache-2.0). The embedding model (BGE Small), reranker (MiniLM Cross-Encoder), generator LLM (Gemma-4), FAISS vector index, and supporting libraries are therefore suitable for commercial deployment.
