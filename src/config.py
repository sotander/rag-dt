# src/config.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

LLM_MODEL_NAME = "google/gemma-4-E4B-it"

OVERLAP = 64
MAX_INPUT_TOKENS = 6000

GOOGLE_DRIVE_FILE_ID = "1cyAa31NYKc3-dFMT1OXLq5bY4Kx8b7uS"

DATA_DIR = "data"
