import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

CHROMA_COLLECTION = "treaty_chunks"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_LENGTH", "8192"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))
